import axios from "axios";
import type { Job, ModelEntry, Dataset, HFSearchResult, SystemStats } from "@/types";

const http = axios.create({ baseURL: "/api" });

// Jobs
export const getJobs = () => http.get<Job[]>("/jobs").then((r) => r.data);
export const getJob = (id: number) => http.get<Job>(`/jobs/${id}`).then((r) => r.data);
export const createJob = (body: {
  name: string;
  training_method: string;
  peft_method: string;
  model_id?: number;
  dataset_id?: number;
  config: Record<string, unknown>;
}) => http.post<Job>("/jobs", body).then((r) => r.data);
export const cancelJob = (id: number) => http.delete(`/jobs/${id}`);
export const updateJobRemarks = (id: number, remarks: string) =>
  http.patch<Job>(`/jobs/${id}/remarks`, { remarks }).then((r) => r.data);
export const getJobMetrics = (id: number) =>
  http.get<{ id: number; step: number; epoch: number | null; loss: number | null; eval_loss: number | null; learning_rate: number | null; reward: number | null; grad_norm: number | null }[]>(`/jobs/${id}/metrics/all`).then((r) => r.data);

// Models
export const getModels = () => http.get<ModelEntry[]>("/models").then((r) => r.data);
export const registerModel = (body: { name: string; hf_repo: string; architecture?: string; template?: string }) =>
  http.post<ModelEntry>("/models", body).then((r) => r.data);
export const downloadModel = (id: number) => http.post(`/models/${id}/download`);
export const getModelDownloadStatus = (id: number) =>
  http.get<{ is_downloaded: boolean; local_path: string | null; downloaded_at: string | null }>(`/models/${id}/download-status`).then((r) => r.data);
export const deleteModel = (id: number) => http.delete(`/models/${id}`);
export const searchHub = (q: string) =>
  http.get<HFSearchResult[]>("/models/search/hub", { params: { q } }).then((r) => r.data);

// Datasets
export const getDatasets = () => http.get<Dataset[]>("/datasets").then((r) => r.data);
export const uploadDataset = (form: FormData) =>
  http.post<Dataset>("/datasets", form, { headers: { "Content-Type": "multipart/form-data" } }).then((r) => r.data);
export const deleteDataset = (id: number) => http.delete(`/datasets/${id}`);

// Exports
export const exportJob = (jobId: number, output_name?: string) =>
  http.post(`/exports/${jobId}`, { output_name }).then((r) => r.data);
export const exportFromPath = (body: { adapter_path: string; output_name?: string }) =>
  http.post("/exports/from-path", body).then((r) => r.data);
export const getExports = () => http.get<{ name: string; path: string; size_mb: number; created_at: string }[]>("/exports").then((r) => r.data);

// Eval
export const startEval = (body: Record<string, unknown>) =>
  http.post<{ run_id: string }>("/eval/run", body).then((r) => r.data);
export const getEvalResult = (runId: string) =>
  http.get<{ status: string; loss?: number; perplexity?: number; output_file?: string; error?: string }>(`/eval/${runId}/result`).then((r) => r.data);

// Chat
export const loadChatModel = (body: { model_path: string; adapter_path?: string; quantization?: string }) =>
  http.post("/chat/load", body).then((r) => r.data);
export const getChatStatus = () =>
  http.get<{ status: string; model_path: string | null; adapter_path: string | null; error: string | null }>("/chat/status").then((r) => r.data);
export const unloadChatModel = () => http.post("/chat/unload").then((r) => r.data);

// System
export const getSystemStats = () => http.get<SystemStats>("/system").then((r) => r.data);

// ASR
export const getASRModels = () => http.get<{ id: string; params: string }[]>("/asr/models").then((r) => r.data);
export const getASRDatasets = () => http.get<Dataset[]>("/asr/datasets").then((r) => r.data);
export const uploadASRDataset = (form: FormData) =>
  http.post<Dataset>("/asr/datasets", form, { headers: { "Content-Type": "multipart/form-data" } }).then((r) => r.data);
export const deleteASRDataset = (id: number) => http.delete(`/asr/datasets/${id}`);
export const uploadASRZip = (form: FormData) =>
  http.post<Dataset>("/asr/datasets/zip", form, { headers: { "Content-Type": "multipart/form-data" } }).then((r) => r.data);
export const getASRJobs = () => http.get<Job[]>("/asr/jobs").then((r) => r.data);
export const getASRJob = (id: number) => http.get<Job>(`/asr/jobs/${id}`).then((r) => r.data);
export const createASRJob = (body: {
  name: string;
  peft_method: string;
  dataset_id?: number;
  val_dataset_id?: number;
  config: Record<string, unknown>;
}) => http.post<Job>("/asr/jobs", body).then((r) => r.data);
export const cancelASRJob = (id: number) => http.delete(`/asr/jobs/${id}`);
