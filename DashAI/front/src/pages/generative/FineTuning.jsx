import {
  Alert,
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  Stack,
  Step,
  StepLabel,
  Stepper,
  Tab,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import FormSchemaLayout from "../../components/shared/FormSchemaLayout";
import { getDatasets, getDatasetSample } from "../../api/datasets";
import {
  cancelFineTuningRun,
  createFineTuningRun,
  deleteFineTuningRun,
  deleteLocalModel,
  getFineTuningCatalog,
  getFineTuningRuns,
  getLocalModels,
  preflightFineTuning,
  startFineTuningRun,
} from "../../api/fineTuning";
import { createGenerativeSession } from "../../api/generativeTask";

const copy = {
  en: {
    title: "Local LLM fine-tuning",
    subtitle: "Train a LoRA/QLoRA adapter and use it directly in Generative.",
    runs: "Runs",
    inventory: "Local inventory",
    newRun: "New run",
    model: "Model and hardware",
    dataset: "Dataset mapping",
    configuration: "Configuration",
    review: "Review and run",
    preflight: "Run preflight",
    create: "Create and start",
    open: "Open in Generative",
    cancel: "Cancel",
    delete: "Delete",
    noRuns: "No fine-tuning runs yet.",
  },
  es: {
    title: "Fine-tuning local de LLM",
    subtitle: "Entrena un adaptador LoRA/QLoRA y úsalo en Generative.",
    runs: "Ejecuciones",
    inventory: "Inventario local",
    newRun: "Nueva ejecución",
    model: "Modelo y hardware",
    dataset: "Mapeo del dataset",
    configuration: "Configuración",
    review: "Revisión y ejecución",
    preflight: "Ejecutar diagnóstico",
    create: "Crear e iniciar",
    open: "Abrir en Generative",
    cancel: "Cancelar",
    delete: "Eliminar",
    noRuns: "Aún no hay ejecuciones de fine-tuning.",
  },
};

const quickParameters = {
  preset: "quick_test",
  max_samples: 32,
  max_steps: 3,
  num_train_epochs: 1,
  max_length: 128,
  per_device_train_batch_size: 1,
  gradient_accumulation_steps: 1,
  learning_rate: 0.0002,
  lora_r: 8,
  lora_alpha: 16,
  lora_dropout: 0.05,
  target_modules: "all-linear",
  seed: 42,
  logging_steps: 1,
  save_steps: 1,
  eval_steps: 1,
};

const emptyDraft = {
  name: `Fine-tuning ${new Date().toLocaleDateString()}`,
  dataset_id: "",
  base_model_id: "qwen2.5-0.5b-instruct",
  base_model_revision: "main",
  method: "qlora",
  dataset_mapping: {
    format: "prompt_completion",
    prompt_column: "",
    completion_column: "",
    validation_split: 0.1,
  },
  training_parameters: quickParameters,
};

function errorMessage(error) {
  return error?.response?.data?.detail || error?.message || String(error);
}

function bytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const order = Math.min(Math.floor(Math.log(value) / Math.log(1024)), 3);
  return `${(value / 1024 ** order).toFixed(1)} ${units[order]}`;
}

export default function FineTuning() {
  const navigate = useNavigate();
  const { i18n } = useTranslation();
  const lang = i18n.language?.split("-")[0] === "es" ? "es" : "en";
  const t = copy[lang];
  const [tab, setTab] = useState(0);
  const [step, setStep] = useState(0);
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [catalog, setCatalog] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [columns, setColumns] = useState([]);
  const [runs, setRuns] = useState([]);
  const [inventory, setInventory] = useState([]);
  const [draft, setDraft] = useState(emptyDraft);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const [nextRuns, nextInventory] = await Promise.all([
      getFineTuningRuns(),
      getLocalModels(),
    ]);
    setRuns(nextRuns || []);
    setInventory(nextInventory || []);
  }, []);

  useEffect(() => {
    Promise.all([getFineTuningCatalog(), getDatasets()])
      .then(([nextCatalog, nextDatasets]) => {
        setCatalog(nextCatalog);
        setDatasets(nextDatasets);
      })
      .catch((nextError) => setError(errorMessage(nextError)));
    refresh().catch((nextError) => setError(errorMessage(nextError)));
  }, [refresh]);

  useEffect(() => {
    if (!runs.some((run) => ["queued", "running"].includes(run.status))) return;
    const timer = window.setInterval(() => {
      refresh().catch(() => {});
    }, 2000);
    return () => window.clearInterval(timer);
  }, [runs, refresh]);

  useEffect(() => {
    if (!draft.dataset_id) {
      setColumns([]);
      return;
    }
    getDatasetSample(Number(draft.dataset_id))
      .then((sample) => {
        const nextColumns = Object.keys(sample);
        setColumns(nextColumns);
        setDraft((current) => ({
          ...current,
          dataset_mapping: {
            ...current.dataset_mapping,
            text_column:
              current.dataset_mapping.text_column || nextColumns[0] || "",
            prompt_column:
              current.dataset_mapping.prompt_column || nextColumns[0] || "",
            completion_column:
              current.dataset_mapping.completion_column || nextColumns[1] || "",
            messages_column:
              current.dataset_mapping.messages_column || nextColumns[0] || "",
          },
        }));
      })
      .catch((nextError) => setError(errorMessage(nextError)));
  }, [draft.dataset_id]);

  const payload = useMemo(
    () => ({ ...draft, dataset_id: Number(draft.dataset_id) }),
    [draft],
  );

  const setMapping = (field, value) =>
    setDraft((current) => ({
      ...current,
      dataset_mapping: { ...current.dataset_mapping, [field]: value },
    }));

  const choosePreset = (preset) => {
    const parameters =
      catalog?.presets?.[preset]?.parameters || quickParameters;
    setDraft((current) => ({
      ...current,
      training_parameters: { ...parameters },
    }));
  };

  const runPreflight = async () => {
    setBusy(true);
    setError("");
    try {
      const result = await preflightFineTuning({
        ...payload,
        download_model: false,
      });
      setReport(result);
      return result;
    } catch (nextError) {
      setError(errorMessage(nextError));
      return null;
    } finally {
      setBusy(false);
    }
  };

  const createAndStart = async () => {
    setBusy(true);
    setError("");
    try {
      const checked = report || (await preflightFineTuning(payload));
      setReport(checked);
      if (!checked.ready) return;
      const created = await createFineTuningRun(payload);
      await startFineTuningRun(created.id);
      await refresh();
      setCreating(false);
      setStep(0);
      setReport(null);
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusy(false);
    }
  };

  const openInGenerative = async (run) => {
    setBusy(true);
    setError("");
    try {
      const session = await createGenerativeSession({
        name: `${run.name} - ${Date.now()}`,
        description: `PEFT adapter from fine-tuning run ${run.id}`,
        task_name: "TextToTextGenerationTask",
        model_name: "PeftAdapterTextGenerationModel",
        fine_tuning_run_id: run.id,
        parameters: {
          max_new_tokens: 128,
          temperature: 0.7,
          top_p: 0.9,
          repetition_penalty: 1.05,
          context_window: 2048,
          device: "auto",
        },
      });
      navigate(`/app/generative/sessions/${session.id}`);
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusy(false);
    }
  };

  const mappingFields = () => {
    const format = draft.dataset_mapping.format;
    const fields =
      format === "text"
        ? [["text_column", "Text"]]
        : format === "messages"
          ? [["messages_column", "Messages"]]
          : [
              ["prompt_column", "Prompt"],
              ["completion_column", "Completion"],
            ];
    return fields.map(([field, label]) => (
      <FormControl fullWidth key={field}>
        <InputLabel>{label}</InputLabel>
        <Select
          label={label}
          value={draft.dataset_mapping[field] || ""}
          onChange={(event) => setMapping(field, event.target.value)}
        >
          {columns.map((column) => (
            <MenuItem value={column} key={column}>
              {column}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    ));
  };

  const renderWizard = () => (
    <Card variant="outlined">
      <CardContent>
        <Stepper activeStep={step} alternativeLabel sx={{ mb: 4 }}>
          {[t.model, t.dataset, t.configuration, t.review].map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        {step === 0 && (
          <Stack spacing={2}>
            <TextField
              label="Name"
              value={draft.name}
              onChange={(event) =>
                setDraft({ ...draft, name: event.target.value })
              }
            />
            <FormControl fullWidth>
              <InputLabel>Base model</InputLabel>
              <Select
                label="Base model"
                value={draft.base_model_id}
                onChange={(event) =>
                  setDraft({ ...draft, base_model_id: event.target.value })
                }
              >
                {(catalog?.models || []).map((model) => (
                  <MenuItem value={model.key} key={model.key}>
                    {model.name} · {model.recommended_vram_gb} GB
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Method</InputLabel>
              <Select
                label="Method"
                value={draft.method}
                onChange={(event) =>
                  setDraft({ ...draft, method: event.target.value })
                }
              >
                <MenuItem value="qlora">QLoRA (NF4)</MenuItem>
                <MenuItem value="lora">LoRA (FP16)</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        )}
        {step === 1 && (
          <Stack spacing={2}>
            <FormControl fullWidth>
              <InputLabel>Dataset</InputLabel>
              <Select
                label="Dataset"
                value={draft.dataset_id}
                onChange={(event) =>
                  setDraft({ ...draft, dataset_id: event.target.value })
                }
              >
                {datasets.map((dataset) => (
                  <MenuItem value={dataset.id} key={dataset.id}>
                    {dataset.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Format</InputLabel>
              <Select
                label="Format"
                value={draft.dataset_mapping.format}
                onChange={(event) => setMapping("format", event.target.value)}
              >
                <MenuItem value="text">text</MenuItem>
                <MenuItem value="prompt_completion">
                  prompt / completion
                </MenuItem>
                <MenuItem value="messages">messages</MenuItem>
              </Select>
            </FormControl>
            {mappingFields()}
          </Stack>
        )}
        {step === 2 && (
          <FormSchemaLayout>
            <Stack spacing={2}>
              <FormControl fullWidth>
                <InputLabel>Preset</InputLabel>
                <Select
                  label="Preset"
                  value={draft.training_parameters.preset}
                  onChange={(event) => choosePreset(event.target.value)}
                >
                  {Object.entries(catalog?.presets || {}).map(
                    ([key, preset]) => (
                      <MenuItem value={key} key={key}>
                        {preset.name}
                      </MenuItem>
                    ),
                  )}
                </Select>
              </FormControl>
              {[
                ["max_samples", "Max samples"],
                ["max_steps", "Max steps"],
                ["max_length", "Sequence length"],
                ["gradient_accumulation_steps", "Gradient accumulation"],
                ["learning_rate", "Learning rate"],
                ["lora_r", "LoRA rank"],
              ].map(([field, label]) => (
                <TextField
                  key={field}
                  type="number"
                  label={label}
                  value={draft.training_parameters[field]}
                  onChange={(event) =>
                    setDraft((current) => ({
                      ...current,
                      training_parameters: {
                        ...current.training_parameters,
                        [field]: Number(event.target.value),
                      },
                    }))
                  }
                />
              ))}
            </Stack>
          </FormSchemaLayout>
        )}
        {step === 3 && (
          <Stack spacing={2}>
            <Typography variant="h6">{draft.name}</Typography>
            <Typography color="text.secondary">
              {draft.base_model_id} · {draft.method.toUpperCase()} · dataset #
              {draft.dataset_id} · {draft.training_parameters.max_steps} steps
            </Typography>
            {report && (
              <>
                <Alert severity={report.ready ? "success" : "error"}>
                  {report.ready ? "Preflight ready" : "Preflight blocked"} ·{" "}
                  {report.hardware.device}
                </Alert>
                {report.blockers.map((issue) => (
                  <Alert severity="error" key={issue.code}>
                    {issue.message}
                  </Alert>
                ))}
                {report.warnings.map((issue) => (
                  <Alert severity="warning" key={issue.code}>
                    {issue.message}
                  </Alert>
                ))}
                <Typography variant="body2">
                  {report.train_rows} train / {report.validation_rows}{" "}
                  validation
                </Typography>
                <Box component="pre" sx={{ overflow: "auto", fontSize: 12 }}>
                  {JSON.stringify(report.preview, null, 2)}
                </Box>
              </>
            )}
            <Stack direction="row" spacing={1}>
              <Button variant="outlined" onClick={runPreflight} disabled={busy}>
                {t.preflight}
              </Button>
              <Button
                variant="contained"
                onClick={createAndStart}
                disabled={busy || (report && !report.ready)}
              >
                {t.create}
              </Button>
            </Stack>
          </Stack>
        )}
      </CardContent>
      <CardActions sx={{ justifyContent: "space-between" }}>
        <Button
          disabled={step === 0}
          onClick={() => setStep((current) => current - 1)}
        >
          Back
        </Button>
        {step < 3 && (
          <Button
            variant="contained"
            disabled={step === 1 && !draft.dataset_id}
            onClick={() => setStep((current) => current + 1)}
          >
            Next
          </Button>
        )}
      </CardActions>
    </Card>
  );

  const renderRuns = () => (
    <Stack spacing={2}>
      {!runs.length && <Alert severity="info">{t.noRuns}</Alert>}
      {runs.map((run) => (
        <Card variant="outlined" key={run.id}>
          <CardContent>
            <Stack direction="row" justifyContent="space-between" spacing={2}>
              <Box>
                <Typography variant="h6">{run.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {run.base_model_id} · {run.method.toUpperCase()} · dataset #
                  {run.dataset_id}
                </Typography>
              </Box>
              <Chip
                label={run.status}
                color={run.status === "failed" ? "error" : "default"}
              />
            </Stack>
            {run.status === "running" || run.status === "queued" ? (
              <Box sx={{ mt: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={run.progress * 100}
                />
                <Typography variant="caption">
                  {run.progress_message}
                </Typography>
              </Box>
            ) : null}
            {run.error_message && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {run.error_message}
              </Alert>
            )}
            {run.metrics && (
              <Typography component="pre" variant="caption">
                {JSON.stringify(run.metrics, null, 2)}
              </Typography>
            )}
          </CardContent>
          <CardActions>
            {run.status === "completed" && (
              <Button
                startIcon={<PlayArrowIcon />}
                onClick={() => openInGenerative(run)}
              >
                {t.open}
              </Button>
            )}
            {["queued", "running"].includes(run.status) && (
              <Button
                color="warning"
                startIcon={<StopIcon />}
                onClick={async () => {
                  await cancelFineTuningRun(run.id);
                  await refresh();
                }}
              >
                {t.cancel}
              </Button>
            )}
            {!["queued", "running"].includes(run.status) && (
              <Button
                color="error"
                startIcon={<DeleteOutlineIcon />}
                onClick={async () => {
                  if (!window.confirm(`${t.delete} ${run.name}?`)) return;
                  try {
                    await deleteFineTuningRun(run.id);
                    await refresh();
                  } catch (nextError) {
                    setError(errorMessage(nextError));
                  }
                }}
              >
                {t.delete}
              </Button>
            )}
          </CardActions>
        </Card>
      ))}
    </Stack>
  );

  const renderInventory = () => (
    <Grid container spacing={2}>
      {inventory.map((item) => (
        <Grid size={{ xs: 12, md: 6 }} key={item.key}>
          <Card variant="outlined">
            <CardContent>
              <Stack direction="row" justifyContent="space-between">
                <Typography variant="h6">{item.name}</Typography>
                <Chip label={item.kind} size="small" />
              </Stack>
              <Typography variant="body2">{item.source}</Typography>
              <Typography variant="caption" sx={{ wordBreak: "break-all" }}>
                {item.path}
              </Typography>
              <Typography>{bytes(item.size_bytes)}</Typography>
            </CardContent>
            {item.kind === "base" && (
              <CardActions>
                <Button
                  color="error"
                  disabled={item.in_use}
                  onClick={async () => {
                    if (!window.confirm(`${t.delete} ${item.name}?`)) return;
                    try {
                      await deleteLocalModel(item.key);
                      await refresh();
                    } catch (nextError) {
                      setError(errorMessage(nextError));
                    }
                  }}
                >
                  {t.delete}
                </Button>
              </CardActions>
            )}
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 1 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate("/app/generative")}
        >
          Generative
        </Button>
        <Box flex={1}>
          <Typography variant="h4">{t.title}</Typography>
          <Typography color="text.secondary">{t.subtitle}</Typography>
        </Box>
        <Button
          variant="contained"
          onClick={() => {
            setCreating(true);
            setTab(0);
          }}
        >
          {t.newRun}
        </Button>
      </Stack>
      <Divider sx={{ mb: 2 }} />
      {error && (
        <Alert severity="error" onClose={() => setError("")} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {busy && <LinearProgress sx={{ mb: 2 }} />}
      {creating ? (
        renderWizard()
      ) : (
        <>
          <Tabs
            value={tab}
            onChange={(_, value) => setTab(value)}
            sx={{ mb: 2 }}
          >
            <Tab label={t.runs} />
            <Tab label={t.inventory} />
          </Tabs>
          {!catalog ? (
            <CircularProgress />
          ) : tab === 0 ? (
            renderRuns()
          ) : (
            renderInventory()
          )}
        </>
      )}
    </Container>
  );
}
