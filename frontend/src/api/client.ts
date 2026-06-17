import axios from "axios";
import { ChatResponse } from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const checkHealth = async (): Promise<boolean> => {
  try {
    // Attempt to hit the root endpoint first
    await apiClient.get("/");
    return true;
  } catch (error: any) {
    // A 404 on the root implies the server is up but doesn't serve a root path.
    if (error.response && error.response.status === 404) {
      return true;
    }
    
    // Fallback: send an OPTIONS request to the known /api/chat endpoint
    try {
      await apiClient.options("/api/chat");
      return true;
    } catch (fallbackError) {
      return false;
    }
  }
};

export const sendMessage = async (message: string): Promise<ChatResponse> => {
  const response = await apiClient.post<ChatResponse>("/api/chat", {
    message,
    session_id: crypto.randomUUID(),
  });
  return response.data;
};
