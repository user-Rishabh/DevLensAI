import React, { useState } from 'react';
import {
  Sparkles,
  Layers,
  Cpu,
  Database,
  Globe,
  RefreshCw,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Target,
  Users,
  Code2,
  ArrowRight,
  ShieldCheck,
  Bot
} from 'lucide-react';

export interface RepoSummaryData {
  tagline?: string;
  overview?: string;
  core_purpose?: string;
  target_audience?: string;
  key_features?: string[];
  tech_stack?: {
    frontend?: string[];
    backend?: string[];
    database_storage?: string[];
    ai_services?: string[];
    languages?: string[];
    frameworks?: string[];
    storage?: string[];
  };
  architecture_summary?: string;
  model_used?: string;
  repo_name?: string;
  generated_at?: number;
}

interface RepoSummaryCardProps {
  summary: RepoSummaryData | null;
  loading: boolean;
  onRegenerate: () => void;
  repoName: string;
}

export default function RepoSummaryCard({
  summary,
  loading,
  onRegenerate,
  repoName
}: RepoSummaryCardProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (loading) {
    return (
      <div className="mb-8 p-6 rounded-2xl bg-white border border-[#E5E1DA] border-t-[4px] border-t-[#E85D2C] shadow-sm animate-pulse">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#E85D2C]/15 flex items-center justify-center text-[#E85D2C]">
              <Sparkles className="w-4 h-4 animate-spin text-[#E85D2C]" />
            </div>
            <div>
              <div className="h-4 w-48 bg-[#E5E1DA] rounded mb-1.5" />
              <div className="h-3 w-32 bg-[#E5E1DA]/60 rounded" />
            </div>
          </div>
          <div className="h-6 w-36 bg-[#E85D2C]/10 rounded-full" />
        </div>
        <div className="space-y-3 pt-2">
          <div className="h-3.5 bg-[#E5E1DA]/70 rounded w-full" />
          <div className="h-3.5 bg-[#E5E1DA]/70 rounded w-5/6" />
          <div className="h-3.5 bg-[#E5E1DA]/70 rounded w-4/6" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-5 pt-4 border-t border-[#E5E1DA]">
          <div className="h-16 bg-[#FAFAF7] rounded-xl border border-[#E5E1DA]" />
          <div className="h-16 bg-[#FAFAF7] rounded-xl border border-[#E5E1DA]" />
          <div className="h-16 bg-[#FAFAF7] rounded-xl border border-[#E5E1DA]" />
        </div>
      </div>
    );
  }

  if (!summary) {
    return null;
  }

  const isGemini = (summary.model_used || '').toLowerCase().includes('gemini');
  const techStack = summary.tech_stack || {};

  return (
    <div className="mb-8 p-6 sm:p-7 rounded-2xl bg-white border border-[#E5E1DA] border-t-[4px] border-t-[#E85D2C] shadow-sm animate-fade-in-stagger transition-all duration-200">
      {/* Header bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-[#E85D2C]/15 border border-[#E85D2C]/30 flex items-center justify-center text-[#E85D2C] shadow-2xs shrink-0">
            <Sparkles className="w-5 h-5 text-[#E85D2C]" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-mono font-bold text-[#E85D2C] uppercase tracking-wider">
                Executive Repository Summary
              </span>
              <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10.5px] font-mono font-semibold border ${
                isGemini
                  ? 'bg-blue-50 text-blue-700 border-blue-200'
                  : 'bg-emerald-50 text-emerald-700 border-emerald-200'
              }`}>
                <Bot className="w-3 h-3" />
                {isGemini ? 'Powered by Gemini 2.0 Flash' : `AI Synthesized (${summary.model_used || 'DevLens Engine'})`}
              </span>
            </div>
            <h2 className="text-lg font-bold text-[#1A1D29] tracking-tight mt-0.5" style={{ fontFamily: '"Space Grotesk", sans-serif' }}>
              What is this website & codebase about?
            </h2>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-center">
          <button
            onClick={onRegenerate}
            title="Regenerate whole repository summary using Gemini 2.0 Flash"
            className="text-xs font-mono text-[#6B7280] hover:text-[#1A1D29] bg-[#FAFAF7] hover:bg-[#E5E1DA]/50 border border-[#E5E1DA] px-2.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all duration-150 cursor-pointer shadow-2xs font-medium"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[#E85D2C]" />
            <span className="hidden sm:inline">Regenerate</span>
          </button>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs font-mono text-[#6B7280] hover:text-[#1A1D29] bg-[#FAFAF7] hover:bg-[#E5E1DA]/50 border border-[#E5E1DA] px-2.5 py-1.5 rounded-lg flex items-center gap-1 transition-all duration-150 cursor-pointer shadow-2xs font-medium"
          >
            {isExpanded ? (
              <>
                <span>Collapse</span>
                <ChevronUp className="w-3.5 h-3.5" />
              </>
            ) : (
              <>
                <span>Expand details</span>
                <ChevronDown className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Tagline */}
      {summary.tagline && (
        <div className="mb-4 pb-3.5 border-b border-[#E5E1DA]/80">
          <p className="text-base sm:text-lg font-semibold text-[#1A1D29] leading-snug">
            "{summary.tagline}"
          </p>
        </div>
      )}

      {/* Detailed Overview: What the website is about */}
      <div className="mb-5">
        <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#6B7280] uppercase tracking-wider mb-2">
          <Globe className="w-3.5 h-3.5 text-[#E85D2C]" />
          <span>Website & Product Purpose</span>
        </div>
        <div className="text-sm text-[#374151] leading-relaxed font-sans space-y-2 whitespace-pre-line bg-[#FAFAF7] p-4 sm:p-5 rounded-xl border border-[#E5E1DA]">
          {summary.overview}
        </div>
      </div>

      {isExpanded && (
        <div className="space-y-5 animate-fade-in-stagger">
          {/* Mission & Target Audience Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {summary.core_purpose && (
              <div className="p-4 rounded-xl bg-white border border-[#E5E1DA] shadow-2xs flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#E85D2C] uppercase tracking-wider mb-1.5">
                    <Target className="w-3.5 h-3.5 text-[#E85D2C]" />
                    <span>Core Value Proposition</span>
                  </div>
                  <p className="text-xs sm:text-sm text-[#4B5563] leading-relaxed">
                    {summary.core_purpose}
                  </p>
                </div>
              </div>
            )}

            {summary.target_audience && (
              <div className="p-4 rounded-xl bg-white border border-[#E5E1DA] shadow-2xs flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#7C3AED] uppercase tracking-wider mb-1.5">
                    <Users className="w-3.5 h-3.5 text-[#7C3AED]" />
                    <span>Target Audience</span>
                  </div>
                  <p className="text-xs sm:text-sm text-[#4B5563] leading-relaxed">
                    {summary.target_audience}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Key Features List */}
          {summary.key_features && summary.key_features.length > 0 && (
            <div>
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#6B7280] uppercase tracking-wider mb-2.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#059669]" />
                <span>Key Platform Capabilities & Features</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {summary.key_features.map((feat, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-2.5 p-3 rounded-xl bg-[#FAFAF7] border border-[#E5E1DA] hover:border-[#E85D2C]/40 transition-colors"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-[#E85D2C] mt-1.5 shrink-0" />
                    <span className="text-xs text-[#374151] leading-relaxed">
                      {feat}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tech Stack & Architecture */}
          <div className="p-4 sm:p-5 rounded-xl bg-[#FAFAF7] border border-[#E5E1DA]">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#6B7280] uppercase tracking-wider mb-3">
              <Layers className="w-3.5 h-3.5 text-[#4F46E5]" />
              <span>Technology & Architecture Blueprint</span>
            </div>

            {/* Tech badges */}
            <div className="flex flex-wrap gap-2 mb-3.5">
              {Object.entries(techStack).map(([category, items]) => {
                if (!Array.isArray(items) || items.length === 0) return null;
                return items.map((tech, i) => (
                  <span
                    key={`${category}-${i}`}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-white border border-[#E5E1DA] text-[11px] font-mono text-[#1A1D29] font-medium shadow-2xs"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-[#E85D2C]" />
                    {tech}
                  </span>
                ));
              })}
            </div>

            {summary.architecture_summary && (
              <p className="text-xs text-[#4B5563] leading-relaxed pt-3 border-t border-[#E5E1DA] font-mono">
                {summary.architecture_summary}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

