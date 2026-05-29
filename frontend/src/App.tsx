/*
NOTE: Redesigned the JSX and component structure of the Research Assistant chat app.
- Added Sidebar component for session management and "New Chat" functionality.
- Implemented responsive hamburger toggle state for mobile layouts.
- Updated messages loop to render custom styled User bubbles and borderless Agent components with a glowing teal avatar indicator.
- Replaced plain tool text with styled Tool Invocation Cards showing status (spinner vs checkmark).
- Added inline drag-and-drop PDF upload cards (Dropzone) and paperclip footer integration.
- Integrated auto-growing textarea heights and Ctrl/Cmd+Enter submission behavior.
- Rendered SSE errors in a slim, dismissable red banner.
*/

import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Bot, 
  Wrench, 
  Menu, 
  Plus, 
  Search, 
  BookOpen, 
  GitMerge, 
  Upload, 
  Check, 
  FileText, 
  AlertCircle, 
  X, 
  Paperclip 
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

type Role = 'user' | 'agent' | 'tool';

interface Message {
  id: string;
  role: Role;
  content: string;
  toolName?: string;
  toolInput?: string;
}

interface DropzoneProps {
  uploadState?: {
    status: 'idle' | 'uploading' | 'success' | 'error';
    filename?: string;
    progress?: number;
    errorMsg?: string;
  };
  onFileSelect: (file: File) => void;
}

function Dropzone({ uploadState, onFileSelect }: DropzoneProps) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf" || file.name.endsWith('.pdf')) {
        onFileSelect(file);
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  if (!uploadState || uploadState.status === 'idle') {
    return (
      <div 
        className={`dropzone-card ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
        role="button"
        tabIndex={0}
        aria-label="Upload PDF file"
      >
        <input 
          ref={fileInputRef} 
          type="file" 
          className="hidden-file-input" 
          style={{ display: 'none' }} 
          accept=".pdf" 
          onChange={handleChange} 
        />
        <Upload className="dropzone-icon" size={24} />
        <p className="dropzone-text">
          Drag & drop PDF here, or <span>browse files</span>
        </p>
        <p className="dropzone-subtext">Supports PDF academic papers up to 25MB</p>
      </div>
    );
  }

  return (
    <div className="dropzone-card success-upload-card" style={{ borderStyle: 'solid' }}>
      <div className="upload-status-container">
        <div className="file-info">
          <FileText size={18} className="tool-icon" />
          <span className="file-name" title={uploadState.filename}>{uploadState.filename}</span>
        </div>
        
        {uploadState.status === 'uploading' && (
          <>
            <div className="upload-progress-bar">
              <div className="upload-progress-fill" style={{ width: '60%' }}></div>
            </div>
            <span className="upload-status-label uploading">Uploading and embedding paper...</span>
          </>
        )}

        {uploadState.status === 'success' && (
          <>
            <span className="upload-status-label success">Successfully Ingested</span>
            <p className="dropzone-subtext">The paper has been chunked and stored in ChromaDB vector space.</p>
          </>
        )}

        {uploadState.status === 'error' && (
          <>
            <span className="upload-status-label error">Ingestion Failed</span>
            <p className="dropzone-subtext" style={{ color: 'var(--error)' }}>{uploadState.errorMsg || 'An unknown error occurred.'}</p>
            <button 
              className="new-chat-btn" 
              style={{ margin: '8px 0 0', padding: '6px 12px' }}
              onClick={(e) => {
                e.stopPropagation();
                if (fileInputRef.current) fileInputRef.current.value = '';
                onButtonClick();
              }}
            >
              Try Again
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// Markdown customizations
const markdownComponents = {
  code({ className, children, ...props }: React.ComponentPropsWithoutRef<'code'>) {
    const match = /language-(\w+)/.exec(className || '');
    const isInline = !match && !String(children).includes('\n');
    return !isInline ? (
      <pre style={{ position: 'relative' }}>
        {match && <span className="code-lang-label">{match[1]}</span>}
        <code className={className} {...props}>
          {children}
        </code>
      </pre>
    ) : (
      <code className={className} {...props}>
        {children}
      </code>
    );
  },
  blockquote({ children, ...props }: React.ComponentPropsWithoutRef<'blockquote'>) {
    return <blockquote {...props}>{children}</blockquote>;
  }
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  // Custom states for redesign
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeToolMsgId, setActiveToolMsgId] = useState<string | null>(null);
  const [errorBannerMsg, setErrorBannerMsg] = useState<string | null>(null);
  const [uploadStates, setUploadStates] = useState<Record<string, {
    status: 'idle' | 'uploading' | 'success' | 'error';
    filename?: string;
    progress?: number;
    errorMsg?: string;
  }>>({});

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleNewChat = () => {
    setMessages([]);
    setSessionId(null);
    setIsLoading(false);
    setActiveToolMsgId(null);
    setErrorBannerMsg(null);
    setIsSidebarOpen(false);
  };

  const getSessionTitle = () => {
    const firstUserMsg = messages.find(m => m.role === 'user');
    if (!firstUserMsg) return 'New Session';
    const text = firstUserMsg.content;
    return text.length > 20 ? text.slice(0, 18) + '...' : text;
  };

  const getToolIcon = (name?: string) => {
    switch (name) {
      case 'research_topic':
        return <Search size={16} className="tool-icon" />;
      case 'literature_review':
        return <BookOpen size={16} className="tool-icon" />;
      case 'citation_graph':
        return <GitMerge size={16} className="tool-icon" />;
      case 'ingest_pdf':
        return <Upload size={16} className="tool-icon" />;
      default:
        return <Wrench size={16} className="tool-icon" />;
    }
  };

  const formatArgs = (args?: string) => {
    if (!args) return '';
    let cleaned = args.trim();
    if (cleaned.startsWith('{') && cleaned.endsWith('}')) {
      try {
        const parsed = JSON.parse(cleaned);
        cleaned = Object.entries(parsed)
          .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
          .join(', ');
      } catch (e) {
        // Fallback to raw text
      }
    }
    if (cleaned.length > 80) {
      return cleaned.slice(0, 77) + '...';
    }
    return cleaned;
  };

  const handleFileUpload = async (msgId: string, file: File) => {
    setUploadStates(prev => ({
      ...prev,
      [msgId]: { status: 'uploading', filename: file.name }
    }));

    try {
      const formData = new FormData();
      formData.append('files', file);

      const response = await fetch('http://localhost:8000/literature/ingest', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed with status ${response.status}`);
      }

      const data = await response.json();
      
      setUploadStates(prev => ({
        ...prev,
        [msgId]: { status: 'success', filename: file.name }
      }));

      console.log("Ingested file successfully:", data);
    } catch (error: any) {
      console.error("Error uploading file:", error);
      setUploadStates(prev => ({
        ...prev,
        [msgId]: { status: 'error', filename: file.name, errorMsg: error.message || 'Upload failed' }
      }));
    }
  };

  const handlePaperclipClick = () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.pdf';
    fileInput.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files;
      if (files && files.length > 0) {
        const file = files[0];
        const uploadMsgId = Date.now().toString() + "-upload";
        
        setMessages(prev => {
          const activeAgentMsg = prev.find(m => m.role === 'agent' && m.content === '' && isLoading);
          const listWithoutAgent = prev.filter(m => m !== activeAgentMsg);
          
          const newToolMsg: Message = { 
            id: uploadMsgId, 
            role: 'tool', 
            toolName: 'ingest_pdf', 
            toolInput: file.name,
            content: `Ingesting ${file.name}` 
          };
          
          const newList = [
            ...listWithoutAgent,
            newToolMsg
          ];
          
          if (activeAgentMsg) {
            newList.push(activeAgentMsg);
          }
          
          return newList;
        });
        
        handleFileUpload(uploadMsgId, file);
      }
    };
    fileInput.click();
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
  };

  useEffect(() => {
    if (input === '') {
      adjustTextareaHeight();
    }
  }, [input]);

  const submitQuery = async (queryText: string) => {
    if (!queryText.trim() || isLoading) return;

    const userMsgId = Date.now().toString();
    setMessages(prev => [...prev, { id: userMsgId, role: 'user', content: queryText }]);
    setInput('');
    setIsLoading(true);
    setErrorBannerMsg(null);
    setActiveToolMsgId(null);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    try {
      const response = await fetch('http://localhost:8000/agent/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: queryText,
          session_id: sessionId 
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      let currentAgentMsgId = Date.now().toString() + "-agent";
      let agentContent = "";
      
      setMessages(prev => [...prev, { id: currentAgentMsgId, role: 'agent', content: '' }]);

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'metadata') {
                  setSessionId(data.session_id);
                } 
                else if (data.type === 'answer_chunk') {
                  agentContent += data.content;
                  setActiveToolMsgId(null);
                  setMessages(prev => 
                    prev.map(msg => 
                      msg.id === currentAgentMsgId 
                        ? { ...msg, content: agentContent }
                        : msg
                    )
                  );
                }
                else if (data.type === 'tool') {
                  const toolMsgId = Date.now().toString() + "-tool-" + data.name;
                  const toolInput = typeof data.input === 'object' 
                    ? JSON.stringify(data.input) 
                    : String(data.input);
                  
                  setActiveToolMsgId(toolMsgId);
                  
                  setMessages(prev => {
                    const withoutAgent = prev.filter(m => m.id !== currentAgentMsgId);
                    return [
                      ...withoutAgent,
                      { id: toolMsgId, role: 'tool', toolName: data.name, toolInput, content: `Invoking ${data.name} with: ${toolInput}` },
                      { id: currentAgentMsgId, role: 'agent', content: agentContent }
                    ];
                  });
                }
                else if (data.type === 'error') {
                  setErrorBannerMsg(data.content);
                }
              } catch (e) {
                console.error("Error parsing SSE JSON", e, line);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("Error fetching from agent:", error);
      setErrorBannerMsg("Sorry, I couldn't connect to the backend server.");
    } finally {
      setIsLoading(false);
      setActiveToolMsgId(null);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitQuery(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      if (e.metaKey || e.ctrlKey) {
        e.preventDefault();
        submitQuery(input);
      }
    }
  };

  const isMac = navigator.userAgent.indexOf('Mac') !== -1;

  return (
    <div className="app-layout">
      {/* Sidebar Backdrop for Mobile */}
      {isSidebarOpen && (
        <div 
          className="sidebar-backdrop" 
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Slim Sidebar Panel */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <Bot size={20} className="tool-icon" />
          <span className="sidebar-title">Research Hub</span>
        </div>
        
        <button 
          className="new-chat-btn" 
          onClick={handleNewChat}
          aria-label="Start a new research chat"
        >
          <Plus size={16} />
          <span>New Chat</span>
        </button>
        
        <div className="sidebar-content">
          <div className="sidebar-section-title">Active Threads</div>
          <ul className="session-list">
            <li 
              className={`session-item ${messages.length > 0 ? 'active' : ''}`}
              onClick={() => setIsSidebarOpen(false)}
            >
              <FileText size={14} />
              <span>{getSessionTitle()}</span>
            </li>
          </ul>
        </div>
        
        <div className="sidebar-footer">
          <span>v1.0.0</span>
          <span>Llama 3.3 70B</span>
        </div>
      </aside>

      {/* Main Workspace Column */}
      <main className="main-content">
        <header className="header">
          <div className="header-left">
            <button 
              className="menu-toggle" 
              onClick={() => setIsSidebarOpen(true)}
              aria-label="Open sidebar"
            >
              <Menu size={20} />
            </button>
            <div className="header-title">
              <Bot size={20} color="var(--accent)" />
              <span>Research Assistant</span>
            </div>
          </div>
          <div className="model-badge">powered by Llama 3.3 70B</div>
        </header>

        {/* Chat Conversation Stream */}
        <div className="chat-container">
          <div className="message-wrapper">
            {messages.length === 0 && (
              <div className="welcome-container">
                <Bot className="welcome-logo" size={48} />
                <h2 className="welcome-title">Research Assistant</h2>
                <p className="welcome-tagline">Multi-source AI research, grounded in papers</p>
                
                <div className="prompt-chips">
                  <button 
                    className="prompt-chip" 
                    onClick={() => submitQuery("Summarize recent work on RAG")}
                  >
                    Summarize recent work on RAG
                  </button>
                  <button 
                    className="prompt-chip" 
                    onClick={() => submitQuery("Find papers on diffusion models")}
                  >
                    Find papers on diffusion models
                  </button>
                  <button 
                    className="prompt-chip" 
                    onClick={() => submitQuery("Analyze citation landscape of Transformer")}
                  >
                    Analyze citation landscape of Transformer
                  </button>
                </div>
              </div>
            )}
            
            {messages.map((msg, index) => {
              const isLatest = index === messages.length - 1;
              
              if (msg.role === 'user') {
                const formattedTime = new Date(parseInt(msg.id) || Date.now()).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                return (
                  <div key={msg.id} className="message user">
                    <div className="bubble">
                      {msg.content}
                    </div>
                    <span className="timestamp">{formattedTime}</span>
                  </div>
                );
              }
              
              if (msg.role === 'tool') {
                const isActive = msg.id === activeToolMsgId && isLoading;
                const isPdfIngest = msg.toolName === 'ingest_pdf';
                
                return (
                  <div key={msg.id} className="tool-group">
                    <div className="tool-card">
                      <div className="tool-info">
                        {getToolIcon(msg.toolName)}
                        <span className="tool-name">{msg.toolName}</span>
                        <span className="tool-args" title={msg.toolInput}>
                          {formatArgs(msg.toolInput)}
                        </span>
                      </div>
                      <div className="tool-status">
                        {isActive ? (
                          <div className="spinner" />
                        ) : (
                          <Check size={14} className="tool-check" />
                        )}
                      </div>
                    </div>
                    
                    {isPdfIngest && (
                      <Dropzone 
                        uploadState={uploadStates[msg.id]}
                        onFileSelect={(file) => handleFileUpload(msg.id, file)}
                      />
                    )}
                  </div>
                );
              }

              if (msg.role === 'agent') {
                // If the stream is active but we haven't received any answer chunks yet, render typing dots
                if (msg.content === '' && isLoading && isLatest) {
                  return (
                    <div key={msg.id} className="message agent">
                      <div className="agent-header">
                        <div className="avatar-dot" />
                        <span className="agent-label">Research Assistant</span>
                      </div>
                      <div className="content" style={{ paddingLeft: '16px' }}>
                        <div className="loading-dots">
                          <div className="dot"></div>
                          <div className="dot"></div>
                          <div className="dot"></div>
                        </div>
                      </div>
                    </div>
                  );
                }

                const formattedTime = new Date(parseInt(msg.id) || Date.now()).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

                return (
                  <div key={msg.id} className="message agent">
                    <div className="agent-header">
                      <div className="avatar-dot" />
                      <span className="agent-label">Research Assistant</span>
                    </div>
                    <div className={`content ${isLoading && isLatest ? 'streaming-cursor' : ''}`}>
                      <ReactMarkdown components={markdownComponents}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    <span className="timestamp">{formattedTime}</span>
                  </div>
                );
              }
              
              return null;
            })}
            
            {/* Inline Error Banner */}
            {errorBannerMsg && (
              <div className="error-banner">
                <div className="error-message">
                  <AlertCircle size={16} className="tool-icon" style={{ color: 'var(--error)' }} />
                  <span>{errorBannerMsg}</span>
                </div>
                <button className="error-dismiss" onClick={() => setErrorBannerMsg(null)} aria-label="Dismiss error">
                  <X size={14} />
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Pinned Bottom Input Bar */}
        <div className="input-outer-container">
          <form className="input-area-wrapper" onSubmit={handleSubmit}>
            <div className="input-area">
              <button 
                type="button" 
                className="input-btn"
                onClick={handlePaperclipClick}
                disabled={isLoading}
                aria-label="Upload PDF document"
              >
                <Paperclip size={18} />
              </button>
              
              <textarea
                ref={textareaRef}
                className="input-textarea"
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  adjustTextareaHeight();
                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask a research question or analyze papers..."
                disabled={isLoading}
                rows={1}
              />
              
              <button 
                type="submit" 
                className="input-btn send-btn" 
                disabled={isLoading || !input.trim()}
                aria-label="Send message"
              >
                <Send size={16} />
              </button>
            </div>
            
            <div className="input-footer-bar">
              <span>Press {isMac ? '⌘↵' : 'Ctrl+Enter'} to send</span>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
