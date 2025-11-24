import './StorageChip.css';
import type { StorageInfo } from '../utils/linkAudit';

interface StorageChipProps {
  storage?: StorageInfo;
}

const LABELS: Record<string, { label: string; icon: string }> = {
  minio: { label: 'MinIO', icon: '🗄️' },
  s3: { label: 'S3', icon: '☁️' },
  external: { label: 'External', icon: '🌐' },
};

export default function StorageChip({ storage }: StorageChipProps) {
  if (!storage) return null;

  const meta = LABELS[storage.provider] ?? LABELS.external;

  return (
    <span
      className={`storage-chip storage-chip--${storage.provider}`}
      title={storage.hostname || storage.rawUrl}
      role="note"
    >
      <span className="storage-chip__icon" aria-hidden="true">
        {meta.icon}
      </span>
      <span className="storage-chip__label">{meta.label}</span>
      {storage.bucket && (
        <span className="storage-chip__bucket" aria-label="Bucket">
          {storage.bucket}
        </span>
      )}
    </span>
  );
}


