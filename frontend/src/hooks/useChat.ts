import { useState, useCallback, useEffect } from "react";
import { ChatResponse } from "../types/api";
import { checkHealth, sendMessage as apiSendMessage } from "../api/client";

export function useChat() {
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);

  // Health check on mount
  useEffect(() => {
    let isMounted = true;
    
    const verifyHealth = async () => {
      const connected = await checkHealth();
      if (isMounted) setIsBackendConnected(connected);
    };

    verifyHealth();

    // Optionally poll every 30 seconds
    const interval = setInterval(verifyHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const sendMessage = useCallback(async (message: string) => {
    if (!message.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await apiSendMessage(message);
      setResponse(data);
    } catch (err: any) {
      console.error("Chat error:", err);
      setError(
        err.response?.data?.detail || err.message || "Failed to communicate with the server."
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearResponse = useCallback(() => {
    setResponse(null);
    setError(null);
  }, []);

  return {
    response,
    isLoading,
    error,
    sendMessage,
    clearResponse,
    isBackendConnected,
  };
}
