export type ChartType = "map" | "profile" | "timeseries" | "none";

export interface ChatResponse {
  answer: string;
  chart_type: ChartType;
  float_ids: string[];
  sql_used: string | null;
  confidence: number;
  metadata: Record<string, any> | null;
  visualization_type: string | null;
  visualization_data: any | null;
  chart_title: string | null;
  chart_description: string | null;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
}
