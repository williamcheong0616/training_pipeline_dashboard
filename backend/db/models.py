from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending/running/completed/failed/cancelled
    training_method = Column(String, nullable=False)  # sft/dpo/ppo/rm/kto/orpo/unsupervised
    peft_method = Column(String, default="lora")  # lora/qlora/dora/full
    model_id = Column(Integer, ForeignKey("model_entries.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    config_json = Column(JSON, default={})
    celery_task_id = Column(String, nullable=True)
    output_dir = Column(String, nullable=True)
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    model = relationship("ModelEntry", back_populates="jobs")
    dataset = relationship("Dataset", back_populates="jobs")
    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    step = Column(Integer, nullable=False)
    epoch = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    reward = Column(Float, nullable=True)
    grad_norm = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="metrics")


class ModelEntry(Base):
    __tablename__ = "model_entries"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    hf_repo = Column(String, nullable=False, unique=True)
    local_path = Column(String, nullable=True)
    architecture = Column(String, nullable=True)
    template = Column(String, default="alpaca")
    is_downloaded = Column(String, default="false")
    downloaded_at = Column(DateTime, nullable=True)

    jobs = relationship("Job", back_populates="model")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    format = Column(String, default="alpaca")  # alpaca/sharegpt/custom/plain_text
    num_samples = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    jobs = relationship("Job", back_populates="dataset")
