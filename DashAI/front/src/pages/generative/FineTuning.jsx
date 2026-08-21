import {
  Alert,
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
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
  downloadLocalModel,
  getFineTuningCatalog,
  getFineTuningRuns,
  getLocalModels,
  preflightFineTuning,
  startFineTuningRun,
} from "../../api/fineTuning";
import { createGenerativeSession } from "../../api/generativeTask";

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
  const { t } = useTranslation(["generative", "common"]);
  const [tab, setTab] = useState(0);
  const [step, setStep] = useState(0);
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
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
    const busyRuns = runs.some((run) =>
      ["queued", "running"].includes(run.status),
    );
    const busyDownloads = inventory.some(
      (item) => item.status === "downloading",
    );
    if (!busyRuns && !busyDownloads) return;
    const timer = window.setInterval(() => {
      refresh().catch(() => {});
    }, 2000);
    return () => window.clearInterval(timer);
  }, [runs, inventory, refresh]);

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

  const openBaseInGenerative = async (item) => {
    setBusy(true);
    setError("");
    try {
      let localModelId = item.local_model_id;
      if (!localModelId) {
        // Filesystem snapshots predating tracked downloads are adopted by
        // the download endpoint without re-downloading the weights.
        try {
          localModelId = (await downloadLocalModel(item.key)).local_model_id;
        } catch (nextError) {
          await refresh();
          const current = (await getLocalModels()).find(
            (candidate) => candidate.key === item.key,
          );
          localModelId = current?.local_model_id;
        }
      }
      if (!localModelId) {
        throw new Error("The managed local model is not ready.");
      }
      const session = await createGenerativeSession({
        name: `${item.name} - ${Date.now()}`,
        description: `Managed local model ${item.key}`,
        task_name: "TextToTextGenerationTask",
        model_name: "LocalManagedTextGenerationModel",
        local_model_id: localModelId,
        parameters: {
          max_new_tokens: 128,
          temperature: 0.7,
          top_p: 0.9,
          top_k: 50,
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
        ? [["text_column", t("generative:fineTuning.label.textColumn")]]
        : format === "messages"
          ? [
              [
                "messages_column",
                t("generative:fineTuning.label.messagesColumn"),
              ],
            ]
          : [
              ["prompt_column", t("generative:fineTuning.label.promptColumn")],
              [
                "completion_column",
                t("generative:fineTuning.label.completionColumn"),
              ],
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

  const numberField = (field, label) => (
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
  );

  const renderWizard = () => (
    <Card variant="outlined">
      <CardContent>
        <Stepper activeStep={step} alternativeLabel sx={{ mb: 4 }}>
          {[
            t("generative:fineTuning.label.stepModel"),
            t("generative:fineTuning.label.stepDataset"),
            t("generative:fineTuning.label.stepConfiguration"),
            t("generative:fineTuning.label.stepReview"),
          ].map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        {step === 0 && (
          <Stack spacing={2}>
            <TextField
              label={t("generative:fineTuning.label.name")}
              value={draft.name}
              onChange={(event) =>
                setDraft({ ...draft, name: event.target.value })
              }
            />
            <FormControl fullWidth>
              <InputLabel>
                {t("generative:fineTuning.label.baseModel")}
              </InputLabel>
              <Select
                label={t("generative:fineTuning.label.baseModel")}
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
              <InputLabel>{t("generative:fineTuning.label.method")}</InputLabel>
              <Select
                label={t("generative:fineTuning.label.method")}
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
              <InputLabel>
                {t("generative:fineTuning.label.dataset")}
              </InputLabel>
              <Select
                label={t("generative:fineTuning.label.dataset")}
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
              <InputLabel>{t("generative:fineTuning.label.format")}</InputLabel>
              <Select
                label={t("generative:fineTuning.label.format")}
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
                <InputLabel>
                  {t("generative:fineTuning.label.preset")}
                </InputLabel>
                <Select
                  label={t("generative:fineTuning.label.preset")}
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
              {numberField(
                "max_samples",
                t("generative:fineTuning.label.maxSamples"),
              )}
              {numberField(
                "max_steps",
                t("generative:fineTuning.label.maxSteps"),
              )}
              <Button
                size="small"
                onClick={() => setShowAdvanced((current) => !current)}
              >
                {showAdvanced
                  ? t("generative:fineTuning.button.hideAdvanced")
                  : t("generative:fineTuning.button.showAdvanced")}
              </Button>
              <Collapse in={showAdvanced}>
                <Stack spacing={2}>
                  {numberField(
                    "max_length",
                    t("generative:fineTuning.label.sequenceLength"),
                  )}
                  {numberField(
                    "gradient_accumulation_steps",
                    t("generative:fineTuning.label.gradientAccumulation"),
                  )}
                  {numberField(
                    "learning_rate",
                    t("generative:fineTuning.label.learningRate"),
                  )}
                  {numberField(
                    "lora_r",
                    t("generative:fineTuning.label.loraRank"),
                  )}
                </Stack>
              </Collapse>
            </Stack>
          </FormSchemaLayout>
        )}
        {step === 3 && (
          <Stack spacing={2}>
            <Typography variant="h6">{draft.name}</Typography>
            <Typography color="text.secondary">
              {draft.base_model_id} · {draft.method.toUpperCase()} · dataset #
              {draft.dataset_id} · {draft.training_parameters.max_steps}{" "}
              {t("generative:fineTuning.label.stepsUnit")}
            </Typography>
            {report && (
              <>
                <Alert severity={report.ready ? "success" : "error"}>
                  {report.ready
                    ? t("generative:fineTuning.label.preflightReady")
                    : t("generative:fineTuning.label.preflightBlocked")}{" "}
                  · {report.hardware.device}
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
                  {t("generative:fineTuning.label.rowsSummary", {
                    train: report.train_rows,
                    validation: report.validation_rows,
                  })}
                </Typography>
                <Box component="pre" sx={{ overflow: "auto", fontSize: 12 }}>
                  {JSON.stringify(report.preview, null, 2)}
                </Box>
              </>
            )}
            <Stack direction="row" spacing={1}>
              <Button variant="outlined" onClick={runPreflight} disabled={busy}>
                {t("generative:fineTuning.button.runPreflight")}
              </Button>
              <Button
                variant="contained"
                onClick={createAndStart}
                disabled={busy || (report && !report.ready)}
              >
                {t("generative:fineTuning.button.createAndStart")}
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
          {t("common:back")}
        </Button>
        {step < 3 && (
          <Button
            variant="contained"
            disabled={step === 1 && !draft.dataset_id}
            onClick={() => setStep((current) => current + 1)}
          >
            {t("common:next")}
          </Button>
        )}
      </CardActions>
    </Card>
  );

  const renderRuns = () => (
    <Stack spacing={2}>
      {!runs.length && (
        <Alert severity="info">
          {t("generative:fineTuning.message.noRuns")}
        </Alert>
      )}
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
            {run.metrics?.health_warnings?.length ? (
              <Alert severity="warning" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  {t("generative:fineTuning.message.healthWarnings")}:
                </Typography>
                {run.metrics.health_warnings.map((warning, index) => (
                  <Typography variant="caption" component="div" key={index}>
                    {warning.message}
                  </Typography>
                ))}
              </Alert>
            ) : null}
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
                {t("generative:fineTuning.button.openInGenerative")}
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
                {t("common:cancel")}
              </Button>
            )}
            {!["queued", "running"].includes(run.status) && (
              <Button
                color="error"
                startIcon={<DeleteOutlineIcon />}
                onClick={async () => {
                  if (
                    !window.confirm(
                      t("generative:fineTuning.message.deleteConfirm", {
                        name: run.name,
                      }),
                    )
                  )
                    return;
                  try {
                    await deleteFineTuningRun(run.id);
                    await refresh();
                  } catch (nextError) {
                    setError(errorMessage(nextError));
                  }
                }}
              >
                {t("common:delete")}
              </Button>
            )}
          </CardActions>
        </Card>
      ))}
    </Stack>
  );

  const statusChip = (item) => {
    if (item.kind === "adapter") {
      return <Chip label={item.status} size="small" />;
    }
    const labelKey = {
      not_downloaded: "statusNotDownloaded",
      downloading: "statusDownloading",
      ready: "statusReady",
      error: "statusError",
    }[item.status];
    const label = labelKey
      ? t(`generative:fineTuning.label.${labelKey}`)
      : item.status;
    const color =
      item.status === "ready"
        ? "success"
        : item.status === "error"
          ? "error"
          : item.status === "downloading"
            ? "info"
            : "default";
    return <Chip label={label} size="small" color={color} />;
  };

  const renderInventory = () => (
    <Grid container spacing={2}>
      {inventory.map((item) => (
        <Grid size={{ xs: 12, md: 6 }} key={item.key}>
          <Card variant="outlined">
            <CardContent>
              <Stack direction="row" justifyContent="space-between">
                <Typography variant="h6">{item.name}</Typography>
                {statusChip(item)}
              </Stack>
              <Typography variant="body2">{item.source}</Typography>
              {item.recommended_vram_gb ? (
                <Typography variant="body2" color="text.secondary">
                  {t("generative:fineTuning.label.recommendedVram", {
                    gb: item.recommended_vram_gb,
                  })}
                </Typography>
              ) : null}
              {item.path ? (
                <Typography variant="caption" sx={{ wordBreak: "break-all" }}>
                  {item.path}
                </Typography>
              ) : null}
              {item.size_bytes ? (
                <Typography>{bytes(item.size_bytes)}</Typography>
              ) : null}
            </CardContent>
            {item.kind === "base" && (
              <CardActions>
                {item.status === "ready" && (
                  <Button
                    startIcon={<PlayArrowIcon />}
                    disabled={busy}
                    onClick={() => openBaseInGenerative(item)}
                  >
                    {t("generative:fineTuning.button.openInGenerative")}
                  </Button>
                )}
                {item.downloadable && (
                  <Button
                    variant={item.status === "error" ? "outlined" : "contained"}
                    disabled={busy}
                    onClick={async () => {
                      try {
                        await downloadLocalModel(item.key);
                        await refresh();
                      } catch (nextError) {
                        setError(errorMessage(nextError));
                      }
                    }}
                  >
                    {item.status === "error"
                      ? t("generative:fineTuning.button.retryDownload")
                      : t("generative:fineTuning.button.download")}
                  </Button>
                )}
                {item.status === "ready" && (
                  <Button
                    color="error"
                    disabled={item.in_use}
                    onClick={async () => {
                      if (
                        !window.confirm(
                          t("generative:fineTuning.message.deleteConfirm", {
                            name: item.name,
                          }),
                        )
                      )
                        return;
                      try {
                        await deleteLocalModel(item.key);
                        await refresh();
                      } catch (nextError) {
                        setError(errorMessage(nextError));
                      }
                    }}
                  >
                    {t("common:delete")}
                  </Button>
                )}
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
          {t("generative:label.generativeModule")}
        </Button>
        <Box flex={1}>
          <Typography variant="h4">
            {t("generative:fineTuning.title")}
          </Typography>
          <Typography color="text.secondary">
            {t("generative:fineTuning.subtitle")}
          </Typography>
        </Box>
        <Button
          variant="contained"
          onClick={() => {
            setCreating(true);
            setTab(0);
          }}
        >
          {t("generative:fineTuning.button.newRun")}
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
            <Tab label={t("generative:fineTuning.label.runs")} />
            <Tab label={t("generative:fineTuning.label.inventory")} />
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
