import api from "./api";
import type {
  FineTuningCatalog,
  FineTuningRun,
  LocalModelInfo,
  PreflightReport,
} from "../types/fineTuning";

const endpoint = "/v1/fine-tuning";

export const getFineTuningCatalog = async (): Promise<FineTuningCatalog> =>
  (await api.get<FineTuningCatalog>(`${endpoint}/catalog`)).data;

export const getFineTuningRuns = async (): Promise<FineTuningRun[]> =>
  (await api.get<FineTuningRun[]>(`${endpoint}/runs`)).data;

export const preflightFineTuning = async (
  payload: Record<string, any>,
): Promise<PreflightReport> =>
  (await api.post<PreflightReport>(`${endpoint}/preflight`, payload)).data;

export const createFineTuningRun = async (
  payload: Record<string, any>,
): Promise<FineTuningRun> =>
  (await api.post<FineTuningRun>(`${endpoint}/runs`, payload)).data;

export const startFineTuningRun = async (id: number): Promise<FineTuningRun> =>
  (await api.post<FineTuningRun>(`${endpoint}/runs/${id}/start`)).data;

export const cancelFineTuningRun = async (id: number): Promise<FineTuningRun> =>
  (await api.post<FineTuningRun>(`${endpoint}/runs/${id}/cancel`)).data;

export const deleteFineTuningRun = async (id: number): Promise<void> => {
  await api.delete(`${endpoint}/runs/${id}`);
};

export const getLocalModels = async (): Promise<LocalModelInfo[]> =>
  (await api.get<LocalModelInfo[]>(`${endpoint}/models`)).data;

export const deleteLocalModel = async (key: string): Promise<void> => {
  await api.delete(`${endpoint}/models/${encodeURIComponent(key)}`);
};

export const downloadLocalModel = async (
  key: string,
): Promise<{ detail: string; local_model_id: number }> =>
  (
    await api.post<{ detail: string; local_model_id: number }>(
      `${endpoint}/models/${encodeURIComponent(key)}/download`,
    )
  ).data;
