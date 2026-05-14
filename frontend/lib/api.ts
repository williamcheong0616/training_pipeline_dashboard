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

// Models
export const getModels = () => http.get<ModelEntry[]>("/models").then((r) => r.data);
export const registerModel = (body: { name: string; hf_repo: string; architecture?: string; template?: string }) =>
  http.post<ModelEntry>("/models", body).then((r) => r.data);
export const downloadModel = (id: number) => http.post(`/models/${id}/download`);
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

// System
export const getSystemStats = () => http.get<SystemStats>("/system").then((r) => r.data);
