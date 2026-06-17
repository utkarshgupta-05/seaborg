# Phase 6 — React Frontend

This document details the architecture and implementation of the SeaBorg **React Frontend** (Phase 6).

## 1. Overview
The frontend is a modern, responsive, scientific-dashboard built with **React**, **Vite**, and **TypeScript**. It acts as the presentation layer for the SeaBorg application, consuming the FastAPI backend built in previous phases. It adheres strictly to the existing backend contracts, reading data dynamically and rendering React-ready JSON payloads natively via Plotly.

## 2. Tech Stack
* **Framework:** React + Vite
* **Language:** TypeScript for strict type-checking against backend Pydantic models.
* **HTTP Client:** Axios
* **Visualization:** Plotly.js (full package to support `scattergeo` map plots) + `react-plotly.js` wrapper.
* **Icons:** React Icons (`react-icons/fa`)
* **Styling:** Vanilla CSS with custom CSS variables for easy theming, avoiding heavy UI frameworks to keep the bundle minimal.

## 3. Architecture & Key Features

### 3.1. API & Type Safety
The `src/types/api.ts` file explicitly defines the `ChatResponse` type matching the FastAPI `models.py`. This ensures type safety when accessing fields like `answer`, `metadata`, and `visualization_data`. The API layer (`src/api/client.ts`) centralizes Axios calls and implements a lightweight backend health check (`GET /`).

### 3.2. Layout Design
The layout uses CSS Grid and Flexbox (in `src/styles/globals.css`) for a responsive, two-pane design at desktop widths (>768px):
* **Sidebar:** Contains the "New Query" button, saved Query History, and an "About" section.
* **Main Area:** Features the `SearchBar`, quick-action example query pills, and the two main content panels: `AnswerPanel` and `ChartPanel`.

On mobile (<768px), the layout elegantly collapses into a vertically stacked flow.

### 3.3. Key Components

* **`Header.tsx`**: Includes the application branding and a live indicator (🟢 / 🔴) displaying the backend's connectivity status.
* **`SearchBar.tsx`**: Handles user input and displays quick "pill" buttons below the input for example queries (e.g., "Temperature at 500m"). This accelerates demos and testing.
* **`QueryHistory.tsx`**: Persists the user's latest 10 queries using browser `localStorage`. Clicking any historical query immediately re-runs it.
* **`AnswerPanel.tsx`**: Displays the semantic answer text and dynamically renders the `QUERY METADATA` box. By using `Object.entries()` on the response `metadata` dictionary, it automatically displays new filters or data points (like region, depth, or record counts) without requiring frontend updates if the backend schema changes in the future (e.g., Phase 3B).
* **`PlotlyChart.tsx` & `ChartPanel.tsx`**: `PlotlyChart` acts as an abstraction layer that receives the `visualization_data` payload from the backend. It wraps the `react-plotly.js` component in a try-catch block, ensuring that if the backend ever returns malformed JSON, the UI gracefully falls back rather than crashing. `ChartPanel` handles the presentation layer, displaying chart titles, descriptions, and utility icons.

## 4. Visual Theming
The frontend leverages a set of centralized CSS variables to enforce a clean, scientific aesthetic:
- **`--bg-dark`**: `#0a1628` (Navy blue for the header)
- **`--bg-light`**: `#f0f4f8` (Main page background)
- **`--card-bg`**: `#ffffff` (Clean white panels)
- **`--accent`**: `#0096b4` (Ocean teal/blue for primary actions and titles)

This ensures visual consistency across the dashboard while maintaining high contrast and readability.
