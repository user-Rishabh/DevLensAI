import React from 'react';
import { 
  ShieldCheck, 
  AlertTriangle, 
  Activity, 
  FileCode, 
  RefreshCw, 
  Layers, 
  GitCommit, 
  Copy, 
  Network, 
  Code2,
  Info
} from 'lucide-react';

export interface QualityScoreItem {
  file_path: string;
  composite_score: number;
  churn_score: number;
  size_score: number;
  complexity_score: number;
  risk_level?: 'healthy' | 'review' | 'risk';
}

export interface MetricDetail {
  score: number;
  weight: number;
  measurement: string;
  rating: string;
  reason: string;
  available: boolean;
}

export interface RepositoryHealth {
  score: number;
  rating: string;
  metric_coverage: number;
  metrics: {
    complexity?: MetricDetail;
    file_size?: MetricDetail;
    duplication?: MetricDetail;
    coupling?: MetricDetail;
    churn?: MetricDetail;
  };
  summary_explanation: string;
}

export interface QualityScoreSummary {
  repo_id?: string;
  file_count?: number;
  average_composite_score?: number;
  needs_attention_count?: number;
  riskiest_files?: QualityScoreItem[];
  repository_health?: RepositoryHealth;
  error?: string;
}

interface QualityScoreCardProps {
  summary: QualityScoreSummary | null;
  loading?: boolean;
  onSelectFile?: (path: string) => void;
  onRecompute?: () => void;
}

export default function QualityScoreCard({ summary, loading, onSelectFile, onRecompute }: QualityScoreCardProps) {
  if (loading) {
    return (
      <div className="bg-white border border-[#E5E1DA] rounded-2xl p-6 shadow-xs animate-pulse">
        <div className="flex items-center justify-between mb-4">
          <div className="h-5 w-44 bg-[#FAFAF7] rounded-md"></div>
          <div className="h-6 w-16 bg-[#FAFAF7] rounded-full"></div>
        </div>
        <div className="h-16 bg-[#FAFAF7] rounded-xl mb-4"></div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
          <div className="h-20 bg-[#FAFAF7] rounded-xl"></div>
          <div className="h-20 bg-[#FAFAF7] rounded-xl"></div>
        </div>
        <div className="space-y-2">
          <div className="h-8 bg-[#FAFAF7] rounded-lg"></div>
          <div className="h-8 bg-[#FAFAF7] rounded-lg"></div>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="bg-white border border-[#E5E1DA] border-t-[4px] border-t-[#059669] rounded-2xl p-6 shadow-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-100 border border-emerald-300 flex items-center justify-center text-emerald-700 shadow-2xs">
              <Activity className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <h3 className="font-bold text-[#1A1D29] text-sm">Code Quality & Health Score</h3>
              <p className="text-xs text-[#6B7280] font-mono mt-0.5">Automated cyclomatic complexity and coupling metrics</p>
            </div>
          </div>
          {onRecompute && (
            <button 
              onClick={onRecompute}
              className="px-3 py-1.5 text-xs font-mono font-semibold bg-[#059669] hover:bg-[#047857] text-white rounded-lg transition-all cursor-pointer shadow-xs active:scale-95"
            >
              Analyze Quality
            </button>
          )}
        </div>
        <p className="text-xs text-[#6B7280] font-mono mt-3">No code quality metrics calculated yet.</p>
      </div>
    );
  }

  if (summary.error) {
    return (
      <div className="bg-white border border-rose-200 border-t-[4px] border-t-rose-500 rounded-2xl p-6 shadow-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-rose-100 border border-rose-300 flex items-center justify-center text-rose-700 shadow-2xs">
              <AlertTriangle className="w-5 h-5 text-rose-600" />
            </div>
            <div>
              <h3 className="font-bold text-[#1A1D29] text-sm">Quality Analysis Error</h3>
              <p className="text-xs text-rose-600 font-mono mt-0.5">{summary.error}</p>
            </div>
          </div>
          {onRecompute && (
            <button 
              onClick={onRecompute}
              className="px-3 py-1.5 text-xs font-mono font-semibold bg-[#E85D2C] hover:bg-[#d44d1e] text-white rounded-lg transition-all cursor-pointer shadow-xs active:scale-95 flex items-center gap-1.5"
            >
              <RefreshCw className="w-3 h-3" /> Retry Analysis
            </button>
          )}
        </div>
      </div>
    );
  }

  const rawScore = summary.average_composite_score;
  const hasValidScore = typeof rawScore === 'number' && !isNaN(rawScore);
  const avgScore = hasValidScore ? Math.round(rawScore) : 100;
  const needsAttention = summary.needs_attention_count ?? 0;
  const isHealthy = hasValidScore ? avgScore >= 80 : true;
  const isModerate = hasValidScore ? avgScore >= 55 && avgScore < 80 : false;

  const repoHealth = summary.repository_health;
  const coveragePercent = repoHealth ? Math.round(repoHealth.metric_coverage * 100) : 100;

  const statusColorClass = !hasValidScore
    ? 'text-gray-700 bg-gray-100 border-gray-300'
    : isHealthy 
      ? 'text-emerald-800 bg-emerald-100 border-emerald-300' 
      : isModerate 
        ? 'text-amber-800 bg-amber-100 border-amber-300' 
        : 'text-rose-800 bg-rose-100 border-rose-300';

  const statusText = repoHealth?.rating || (
    !hasValidScore
      ? 'Analysis In Progress'
      : isHealthy 
        ? 'Healthy Codebase' 
        : isModerate 
          ? 'Moderate Maintainability' 
          : 'High Technical Debt'
  );

  return (
    <div className="bg-white border border-[#E5E1DA] border-t-[4px] border-t-[#059669] rounded-2xl p-6 shadow-xs space-y-6">
      
      {/* Primary Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-emerald-100 border border-emerald-300 flex items-center justify-center shadow-2xs">
            {isHealthy ? (
              <ShieldCheck className="w-5 h-5 text-emerald-700" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-amber-700" />
            )}
          </div>
          <div>
            <h3 className="font-bold text-[#1A1D29] text-base">Absolute Repository Health</h3>
            <p className="text-xs text-[#6B7280] font-mono mt-0.5">
              Standardized maintainability metrics calibrated with fixed engineering thresholds
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {onRecompute && (
            <button 
              onClick={onRecompute}
              title="Recompute Quality Scores"
              className="p-1.5 text-[#6B7280] hover:text-[#1A1D29] bg-[#FAFAF7] hover:bg-[#F0ECE1] border border-[#E5E1DA] rounded-lg transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          )}
          <span className={`text-xs font-mono font-bold px-3 py-1 rounded-full border ${statusColorClass}`}>
            {hasValidScore ? `${avgScore}/100` : '--/100'}
          </span>
        </div>
      </div>

      {/* Overview Status Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#FAFAF7] border border-[#E5E1DA] rounded-xl p-4">
        <div>
          <span className="text-[10px] text-[#6B7280] uppercase tracking-wider font-mono font-semibold block">Maintainability Rating</span>
          <span className={`text-sm font-bold ${isHealthy ? 'text-emerald-700' : isModerate ? 'text-amber-700' : 'text-rose-700'}`}>
            {statusText}
          </span>
          {repoHealth?.summary_explanation && (
            <p className="text-xs text-[#6B7280] font-mono mt-1">
              {repoHealth.summary_explanation}
            </p>
          )}
        </div>
        <div className="flex items-center gap-4 border-t sm:border-t-0 sm:border-l border-[#E5E1DA] pt-2 sm:pt-0 sm:pl-4 shrink-0">
          <div>
            <span className="text-[10px] text-[#6B7280] uppercase tracking-wider font-mono font-semibold block">Metric Coverage</span>
            <span className="text-xs font-mono font-bold text-[#1A1D29]">
              {coveragePercent}%
            </span>
          </div>
          <div>
            <span className="text-[10px] text-[#6B7280] uppercase tracking-wider font-mono font-semibold block">Attention Needed</span>
            <span className="text-xs font-mono font-bold text-[#D97706]">
              {needsAttention} {needsAttention === 1 ? 'file' : 'files'}
            </span>
          </div>
        </div>
      </div>

      {/* Metric Breakdown Grid (5 Dimensions) */}
      {repoHealth?.metrics && Object.keys(repoHealth.metrics).length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs font-mono text-[#059669] bg-[#059669]/10 border border-[#059669]/25 px-2.5 py-0.5 rounded-md font-bold uppercase tracking-wider">
              Quality Dimensions & Explainability
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            
            {/* Complexity */}
            {repoHealth.metrics.complexity && (
              <div className="p-3.5 rounded-xl border border-[#E5E1DA] bg-[#FAFAF7]/60 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-bold text-[#1A1D29]">
                      <Code2 className="w-3.5 h-3.5 text-[#059669]" /> Branching Complexity
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                      repoHealth.metrics.complexity.score >= 80 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                      repoHealth.metrics.complexity.score >= 60 ? 'bg-amber-50 text-amber-700 border-amber-200' :
                      'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {repoHealth.metrics.complexity.available ? `${repoHealth.metrics.complexity.score}/100` : 'N/A'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#D97706] font-semibold block">
                    {repoHealth.metrics.complexity.measurement}
                  </span>
                  <p className="text-[11px] text-[#6B7280] font-mono mt-1.5 leading-snug">
                    {repoHealth.metrics.complexity.reason}
                  </p>
                </div>
              </div>
            )}

            {/* File Size & Modularity */}
            {repoHealth.metrics.file_size && (
              <div className="p-3.5 rounded-xl border border-[#E5E1DA] bg-[#FAFAF7]/60 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-bold text-[#1A1D29]">
                      <Layers className="w-3.5 h-3.5 text-[#3B5FE0]" /> Modularity & File Size
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                      repoHealth.metrics.file_size.score >= 80 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                      repoHealth.metrics.file_size.score >= 60 ? 'bg-amber-50 text-amber-700 border-amber-200' :
                      'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {repoHealth.metrics.file_size.available ? `${repoHealth.metrics.file_size.score}/100` : 'N/A'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#3B5FE0] font-semibold block">
                    {repoHealth.metrics.file_size.measurement}
                  </span>
                  <p className="text-[11px] text-[#6B7280] font-mono mt-1.5 leading-snug">
                    {repoHealth.metrics.file_size.reason}
                  </p>
                </div>
              </div>
            )}

            {/* Code Duplication */}
            {repoHealth.metrics.duplication && (
              <div className="p-3.5 rounded-xl border border-[#E5E1DA] bg-[#FAFAF7]/60 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-bold text-[#1A1D29]">
                      <Copy className="w-3.5 h-3.5 text-[#D97706]" /> Code Duplication
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                      repoHealth.metrics.duplication.score >= 80 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                      repoHealth.metrics.duplication.score >= 60 ? 'bg-amber-50 text-amber-700 border-amber-200' :
                      'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {repoHealth.metrics.duplication.available ? `${repoHealth.metrics.duplication.score}/100` : 'N/A'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#059669] font-semibold block">
                    {repoHealth.metrics.duplication.measurement}
                  </span>
                  <p className="text-[11px] text-[#6B7280] font-mono mt-1.5 leading-snug">
                    {repoHealth.metrics.duplication.reason}
                  </p>
                </div>
              </div>
            )}

            {/* Dependency Coupling */}
            {repoHealth.metrics.coupling && (
              <div className="p-3.5 rounded-xl border border-[#E5E1DA] bg-[#FAFAF7]/60 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-bold text-[#1A1D29]">
                      <Network className="w-3.5 h-3.5 text-[#7C3AED]" /> Dependency Coupling
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                      repoHealth.metrics.coupling.score >= 80 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                      repoHealth.metrics.coupling.score >= 60 ? 'bg-amber-50 text-amber-700 border-amber-200' :
                      'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {repoHealth.metrics.coupling.available ? `${repoHealth.metrics.coupling.score}/100` : 'N/A'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#7C3AED] font-semibold block">
                    {repoHealth.metrics.coupling.measurement}
                  </span>
                  <p className="text-[11px] text-[#6B7280] font-mono mt-1.5 leading-snug">
                    {repoHealth.metrics.coupling.reason}
                  </p>
                </div>
              </div>
            )}

            {/* Git Churn Risk */}
            {repoHealth.metrics.churn && (
              <div className="p-3.5 rounded-xl border border-[#E5E1DA] bg-[#FAFAF7]/60 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-bold text-[#1A1D29]">
                      <GitCommit className="w-3.5 h-3.5 text-[#E85D2C]" /> Hotspot Churn Risk
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                      repoHealth.metrics.churn.score >= 80 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                      repoHealth.metrics.churn.score >= 60 ? 'bg-amber-50 text-amber-700 border-amber-200' :
                      'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {repoHealth.metrics.churn.available ? `${repoHealth.metrics.churn.score}/100` : 'N/A'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#E85D2C] font-semibold block">
                    {repoHealth.metrics.churn.measurement}
                  </span>
                  <p className="text-[11px] text-[#6B7280] font-mono mt-1.5 leading-snug">
                    {repoHealth.metrics.churn.reason}
                  </p>
                </div>
              </div>
            )}

          </div>
        </div>
      )}

      {/* Relative File Risk & Riskiest Files Section */}
      <div className="pt-4 border-t border-[#E5E1DA]">
        <div className="flex items-center justify-between mb-2">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-[#1A1D29]">Relative File Risk (Internal Ranking)</span>
              <span className="text-[10px] font-mono text-[#D97706] bg-[#D97706]/10 border border-[#D97706]/25 px-2 py-0.2 rounded font-semibold">
                Percentile Model
              </span>
            </div>
            <p className="text-[11px] text-[#6B7280] font-mono mt-0.5">
              Identifies which files carry the highest complexity, size, and churn relative to others in this project.
            </p>
          </div>
          <span className="text-[10px] text-[#6B7280] font-mono shrink-0">100 = Lowest Risk</span>
        </div>

        {(summary.file_count === 0) ? (
          <p className="text-xs text-[#6B7280] italic py-2">No source code files available to analyze in this repository.</p>
        ) : (!summary.riskiest_files || summary.riskiest_files.length === 0) ? (
          <p className="text-xs text-[#6B7280] italic py-2">No high-risk files identified.</p>
        ) : (
          <div className="space-y-1.5 mt-3">
            {(summary.riskiest_files || []).map((file) => {
              const score = file.composite_score;
              const badgeClass = score >= 70 
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
                : score >= 45 
                  ? 'bg-amber-50 text-amber-700 border-amber-200' 
                  : 'bg-rose-50 text-rose-700 border-rose-200';

              return (
                <div 
                  key={file.file_path}
                  onClick={() => onSelectFile && onSelectFile(file.file_path)}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-[#FAFAF7] hover:bg-white border border-[#E5E1DA] hover:border-[#E85D2C]/40 cursor-pointer transition-all duration-150 group shadow-2xs"
                >
                  <div className="flex items-center gap-2.5 overflow-hidden pr-2">
                    <FileCode className="w-3.5 h-3.5 text-[#6B7280] group-hover:text-[#E85D2C] shrink-0 transition-colors" />
                    <span className="text-xs font-mono text-[#1A1D29] group-hover:text-[#E85D2C] truncate">
                      {file.file_path}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-[10px] text-[#6B7280] font-mono hidden sm:inline">
                      {file.churn_score < 50 ? 'High Churn' : file.complexity_score < 50 ? 'High Complexity' : 'Large File'}
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${badgeClass}`}>
                      {score}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

    </div>
  );
}
