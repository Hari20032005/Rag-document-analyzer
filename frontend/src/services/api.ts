import axios from 'axios';

export interface UploadResponse {
  job_id: string;
  status: 'processing';
}

export interface JobStatusResponse {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  pages_parsed: number;
  total_pages: number;
  chunks_created: number;
  doc_id: string | null;
  error: string | null;
}

export interface DocumentInfo {
  id: string;
  title: string;
  upload_date: string;
  page_count: number;
}

export interface SourceChunk {
  page: number;
  snippet: string;
  chunk_id: string;
  score: number;
}

export interface AskRequest {
  doc_id: string;
  question: string;
  top_k?: number;
  mode?: 'explain' | 'summary' | 'qa';
  temperature?: number;
}

export interface AskResponse {
  answer: string;
  sources: SourceChunk[];
  prompt_used: string;
}

export interface SummarizeResponse {
  summary: string;
  key_concepts: string[];
  sources: SourceChunk[];
  prompt_used: string;
}

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  timeout: 30000,
});
const llmTimeoutMs = Number(import.meta.env.VITE_LLM_TIMEOUT_MS ?? 120000);

export async function uploadPdf(
  file: File,
  title?: string,
  onProgress?: (progress: number) => void,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (title) {
    formData.append('title', title);
  }

  const response = await api.post<UploadResponse>('/api/v1/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (event) => {
      if (!onProgress || !event.total) {
        return;
      }
      const progress = Math.round((event.loaded * 100) / event.total);
      onProgress(progress);
    },
  });
  return response.data;
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await api.get<JobStatusResponse>(`/api/v1/status/${jobId}`);
  return response.data;
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const response = await api.get<{ documents: DocumentInfo[] }>('/api/v1/docs');
  return response.data.documents;
}

export async function deleteDocument(docId: string): Promise<void> {
  await api.delete(`/api/v1/docs/${docId}`);
}

export async function askQuestion(payload: AskRequest): Promise<AskResponse> {
  const response = await api.post<AskResponse>('/api/v1/ask', payload, { timeout: llmTimeoutMs });
  return response.data;
}

export async function explainSection(payload: {
  doc_id: string;
  section: 'introduction' | 'methodology' | 'results' | 'conclusion' | 'custom';
  custom_query?: string;
  top_k?: number;
  temperature?: number;
}): Promise<AskResponse> {
  const response = await api.post<AskResponse>('/api/v1/explain-section', payload, { timeout: llmTimeoutMs });
  return response.data;
}

export async function summarizeDocument(payload: {
  doc_id: string;
  length: 'short' | 'medium' | 'long';
  temperature?: number;
}): Promise<SummarizeResponse> {
  const response = await api.post<SummarizeResponse>('/api/v1/summarize', payload, { timeout: llmTimeoutMs });
  return response.data;
}
