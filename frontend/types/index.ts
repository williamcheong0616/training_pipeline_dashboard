export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface Job {
  id: number;
  name: string;
  status: JobStatus;
  training_method: string;
  peft_method: string;
  model_id: number | null;
  dataset_id: number | null;
  output_dir: string | null;
  error_msg: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainingMetric {
  id: number;
  step: number;
  epoch: number | null;
  loss: number | null;
  eval_loss: number | null;
  learning_rate: number | null;
  reward: number | null;
  grad_norm: number | null;
  timestamp: string;
}

export interface ModelEntry {
  id: number;
  name: string;
  hf_repo: string;
  local_path: string | null;
  architecture: string | null;
  template: string;
  is_downloaded: string;
  downloaded_at: string | null;
}

export interface HFSearchResult {
  model_id: string;
  pipeline_tag: string | null;
  downloads: number | null;
  likes: number | null;
}

export interface Dataset {
  id: number;
  name: string;
  path: string;
  format: string;
  num_samples: number | null;
  description: string | null;
  created_at: string;
}

export interface SystemStats {
  cpu_percent: number;
  ram_total_mb: number;
  ram_used_mb: number;
  disk_total_gb: number;
  disk_used_gb: number;
  cuda_available: boolean;
  gpu: {
    index: number;
    name: string;
    total_mb: number;
    used_mb: number;
    free_mb: number;
  }[];
}
