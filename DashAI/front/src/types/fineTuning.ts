export type FineTuningStatus =
  "not_started" | "queued" | "running" | "completed" | "failed" | "canceled";

export interface DatasetMapping {
  format: "text" | "prompt_completion" | "messages";
  text_column?: string | null;
  prompt_column?: string | null;
  completion_column?: string | null;
  messages_column?: string | null;
  validation_split: number;
}

export interface TrainingParameters {
  preset: "quick_test" | "qlora_8gb" | "lora_small";
  max_samples: number;
  max_steps: number;
  num_train_epochs: number;
  max_length: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  learning_rate: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string;
  seed: number;
  logging_steps: number;
  save_steps: number;
  eval_steps: number;
}

export interface FineTuningRun {
  id: number;
  name: string;
  dataset_id: number;
  base_model_id: string;
  base_model_revision: string;
  resolved_model_revision?: string | null;
  method: "lora" | "qlora";
  dataset_mapping: DatasetMapping;
  training_parameters: TrainingParameters;
  status: FineTuningStatus;
  huey_id?: string | null;
  progress: number;
  progress_message?: string | null;
  metrics?: Record<string, any> | null;
  runtime_metadata?: Record<string, any> | null;
  artifact_path?: string | null;
  error_message?: string | null;
  cancellation_requested: boolean;
  created: string;
}

export interface FineTuningCatalog {
  models: Array<{
    key: string;
    name: string;
    repository: string;
    parameter_count_billions: number;
    recommended_vram_gb: number;
  }>;
  presets: Record<
    string,
    { name: string; description: string; parameters: TrainingParameters }
  >;
  capabilities: Record<string, any>;
}

export interface PreflightReport {
  ready: boolean;
  blockers: Array<{ code: string; message: string }>;
  warnings: Array<{ code: string; message: string }>;
  preview: Array<Record<string, any>>;
  dataset_fingerprint?: string | null;
  dataset_rows: number;
  train_rows: number;
  validation_rows: number;
  model_downloaded: boolean;
  hardware: Record<string, any>;
  dependencies: Record<string, string | null>;
}

export interface LocalModelInfo {
  key: string;
  kind: "base" | "adapter";
  name: string;
  source: string;
  path: string;
  size_bytes: number;
  status: string;
  run_id?: number | null;
  in_use: boolean;
}
