import { fireEvent, render, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import FileUploader from '../components/FileUploader';

const uploadPdfMock = vi.fn();

vi.mock('../services/api', () => ({
  uploadPdf: (...args: unknown[]) => uploadPdfMock(...args),
}));

describe('FileUploader', () => {
  it('uploads PDF and emits job id', async () => {
    uploadPdfMock.mockResolvedValue({ job_id: 'job-123', status: 'processing' });
    const onUploaded = vi.fn();
    const onError = vi.fn();

    const { container } = render(<FileUploader onUploaded={onUploaded} onError={onError} />);

    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['mock pdf'], 'paper.pdf', { type: 'application/pdf' });

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(onUploaded).toHaveBeenCalledWith('job-123');
    });
    expect(onError).not.toHaveBeenCalled();
  });
});
