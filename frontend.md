# Research Assistant - Frontend Current State

The frontend is a single-page web application providing an interactive, real-time chat interface to converse with the AI Research Assistant agent.

---

## 🛠️ Technology Stack
- **Framework**: React 19 (TypeScript)
- **Build Tool**: Vite
- **Styling**: Vanilla CSS (no TailwindCSS or component libraries)
- **Markdown Rendering**: `react-markdown`
- **Icons**: `lucide-react`

---

## 📂 Code Structure & Key Files

### 1. Root Structure (`frontend/`)
- **`index.html`**: Entrypoint HTML page. Mounts the React application structure.
- **`vite.config.ts`**: Vite configuration using the `@vitejs/plugin-react` plugin.
- **`tsconfig.json` & configurations**: TypeScript options.

### 2. Source Files (`frontend/src/`)
- **`main.tsx`**: Renders the React root application inside `<App />` using React 19 concurrent features.
- **`App.tsx`**: Main component that contains:
  - **Message State Management**: Tracks active chat messages (`Message[]`) with user, agent, and tool roles.
  - **SSE Streaming Reader**: Establishes a connection to the backend `/agent/` streaming endpoint. Decodes incoming chunks to dynamically append text and tool invocation logs.
  - **Auto-Scrolling**: Utilizes a `useRef` pointing to a dummy div to smoothly scroll the window to the bottom upon new messages/loading status.
- **`App.css`**: Layout styles for headers, input fields, markdown components, and dot animations.
- **`index.css`**: Global design style setup, configuring typography (Inter-like sans-serif), a dark theme color system, message bubble alignments, and scrollbars.

---

## 🔌 API Integration & Streaming (SSE)

The frontend initiates conversation by making a `POST` request to `http://localhost:8000/agent/`:
- **Payload**:
  ```json
  {
    "query": "user search topic",
    "session_id": "optional-previous-session-id"
  }
  ```
- **SSE Stream Decoding**:
  Uses a `ReadableStreamDefaultReader` on `response.body`. Each chunk is decoded as text, split by newline, and parsed if starting with `data: `:
  - `metadata`: Saves the returned `session_id` to maintain conversation memory in future queries.
  - `answer_chunk`: Appends the next token to the active agent response bubble.
  - `tool`: Inserts a tool execution indicator bubble (`Invoking research_topic with: ...`) directly above the agent response to give the user visibility into the research process.
  - `error`: Formats and renders any server error gracefully.

---

## 🏃 Running the Frontend
To start the development server locally:
```bash
cd frontend
npm run dev
```
By default, this will spin up the server at `http://localhost:5173`.
