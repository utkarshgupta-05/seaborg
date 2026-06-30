import { useState, useCallback, useEffect, useRef } from "react";
import { ChatResponse } from "../types/api";
import { checkHealth, sendMessage as apiSendMessage } from "../api/client";

export type MessagePair = {
  id: string;
  query: string;
  response: ChatResponse | null;
  error: string | null;
  isLoading: boolean;
};

export function useChat() {
  const sessionIdRef = useRef(crypto.randomUUID());
  const [messages, setMessages] = useState<MessagePair[]>([]);
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

    const messageId = crypto.randomUUID();
    const newMessage: MessagePair = {
      id: messageId,
      query: message,
      response: null,
      error: null,
      isLoading: true
    };
    
    setMessages(prev => [...prev, newMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const data = await apiSendMessage(message, sessionIdRef.current);
      setResponse(data);
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { ...msg, response: data, isLoading: false } 
          : msg
      ));
    } catch (err: any) {
      console.error("Chat error:", err);
      const errorMsg = err.response?.data?.detail || err.message || "Failed to communicate with the server.";
      setError(errorMsg);
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { ...msg, error: errorMsg, isLoading: false } 
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearResponse = useCallback(() => {
    setResponse(null);
    setError(null);
    setMessages([]);
    sessionIdRef.current = crypto.randomUUID();
  }, []);

  return {
    response,
    messages,
    isLoading,
    error,
    sendMessage,
    clearResponse,
    isBackendConnected,
  };
}
