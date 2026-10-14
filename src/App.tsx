import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  FileCode,
  FileSpreadsheet,
  Gauge,
  Layers,
  Play,
  RefreshCw,
  Server,
  Shield,
  Sliders,
  Sparkles,
  Terminal,
  Upload,
  Zap,
  Info,
  Check,
  XCircle,
  ExternalLink,
  ChevronDown,
  Download,
  RotateCcw,
  TrendingDown,
  TrendingUp,
  Filter,
  Eye,
  EyeOff,
  SlidersHorizontal,
  HelpCircle,
  Copy,
  X,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

// --- Types & Data Models ---
interface ColumnMeta {
  name: string;
  dtype: string;
  inferredType: 'numerical' | 'categorical' | 'id';
  missingCount: number;
  missingPct: number;
  uniqueCount: number;
  sampleValues: any[];
  stats?: {
    mean: number;
    std: number;
    min: number;
    p25: number;
    p50: number;
    p75: number;
    max: number;
  };
}

interface DatasetItem {
  id: string;
  name: string;
  rowCount: number;
  colCount: number;
  taskType: 'regression' | 'binary_classification';
  targetCol: string;
  columns: ColumnMeta[];
  data: Record<string, any>[];
  warnings: string[];
}

interface ModelItem {
  id: string;
  name: string;
  algorithm: string;
  taskType: 'regression' | 'binary_classification';
  targetCol: string;
  version: number;
  runId: string;
  metrics: Record<string, number>;
  cvMean?: number;
  cvScores?: number[];
  durationSec: number;
  isActive: boolean;
  status: 'validated' | 'provisional';
  createdAt: string;
  activatedAt?: string;
  features: string[];
  referenceDataSummary: Record<string, { mean: number; std: number; median: number }>;
}

interface PredictionLog {
  id: string;
  modelId: string;
  modelVersion: number;
  features: Record<string, any>;
  prediction: any;
  probabilities?: Record<string, number>;
  latencyMs: number;
  timestamp: string;
  actualLabel?: any;
}

interface DriftReport {
  feature: string;
  type: string;
  testMethod: string;
  statistic: number;
  pValue: number;
  driftDetected: boolean;
  refMean: number;
  currMean: number;
  refStd: number;
  currStd: number;
}

interface Alert {
  id: string;
  modelId: string;
  type: 'DATA_DRIFT' | 'HIGH_LATENCY' | 'HIGH_ERROR_RATE';
  severity: 'WARNING' | 'CRITICAL' | 'INFO';
  message: string;
  details: string;
  status: 'ACTIVE' | 'RESOLVED';
  createdAt: string;
}

export interface HistoricalKSEntry {
  time: string;
  timestamp: string;
  batchId: string;
  [key: string]: any;
}

export const FEATURE_COLORS: Record<string, string> = {
  sqft: '#6366f1',           // indigo
  beds: '#06b6d4',           // cyan
  baths: '#f59e0b',          // amber
  year_built: '#a855f7',     // purple
  garage: '#10b981',         // emerald
  tenure_months: '#6366f1',  // indigo
  monthly_charges: '#f43f5e',// rose
  total_charges: '#06b6d4',  // cyan
  support_calls: '#f59e0b',  // amber
};

export const COLOR_PALETTE = ['#6366f1', '#06b6d4', '#f59e0b', '#a855f7', '#10b981', '#ec4899', '#3b82f6'];

export const INITIAL_HOUSING_KS_HISTORY: HistoricalKSEntry[] = [
  { time: '09:00', timestamp: '2026-10-12 09:00', batchId: 'batch_01', sqft: 0.824, beds: 0.891, baths: 0.912, year_built: 0.742, garage: 0.884 },
  { time: '09:30', timestamp: '2026-10-12 09:30', batchId: 'batch_02', sqft: 0.745, beds: 0.852, baths: 0.887, year_built: 0.718, garage: 0.861 },
  { time: '10:00', timestamp: '2026-10-12 10:00', batchId: 'batch_03', sqft: 0.612, beds: 0.795, baths: 0.841, year_built: 0.683, garage: 0.819 },
  { time: '10:30', timestamp: '2026-10-12 10:30', batchId: 'batch_04', sqft: 0.438, beds: 0.741, baths: 0.795, year_built: 0.642, garage: 0.785 },
  { time: '11:00', timestamp: '2026-10-12 11:00', batchId: 'batch_05', sqft: 0.194, beds: 0.682, baths: 0.763, year_built: 0.615, garage: 0.771 },
  { time: '11:30', timestamp: '2026-10-12 11:30', batchId: 'batch_06', sqft: 0.082, beds: 0.651, baths: 0.738, year_built: 0.589, garage: 0.754 },
  { time: '12:00', timestamp: '2026-10-12 12:00', batchId: 'batch_07', sqft: 0.021, beds: 0.638, baths: 0.715, year_built: 0.572, garage: 0.740 },
];

export const INITIAL_CHURN_KS_HISTORY: HistoricalKSEntry[] = [
  { time: '09:00', timestamp: '2026-10-12 09:00', batchId: 'batch_01', tenure_months: 0.842, monthly_charges: 0.789, total_charges: 0.815, support_calls: 0.892 },
  { time: '09:30', timestamp: '2026-10-12 09:30', batchId: 'batch_02', tenure_months: 0.791, monthly_charges: 0.715, total_charges: 0.768, support_calls: 0.841 },
  { time: '10:00', timestamp: '2026-10-12 10:00', batchId: 'batch_03', tenure_months: 0.654, monthly_charges: 0.584, total_charges: 0.692, support_calls: 0.778 },
  { time: '10:30', timestamp: '2026-10-12 10:30', batchId: 'batch_04', tenure_months: 0.482, monthly_charges: 0.392, total_charges: 0.581, support_calls: 0.625 },
  { time: '11:00', timestamp: '2026-10-12 11:00', batchId: 'batch_05', tenure_months: 0.215, monthly_charges: 0.142, total_charges: 0.421, support_calls: 0.312 },
  { time: '11:30', timestamp: '2026-10-12 11:30', batchId: 'batch_06', tenure_months: 0.078, monthly_charges: 0.041, total_charges: 0.285, support_calls: 0.089 },
  { time: '12:00', timestamp: '2026-10-12 12:00', batchId: 'batch_07', tenure_months: 0.0001, monthly_charges: 0.0001, total_charges: 0.006, support_calls: 0.0001 },
];

// --- Sample Datasets ---
const SAMPLE_HOUSING_DATA: Record<string, any>[] = [
  { sqft: 2150, beds: 3, baths: 2.0, year_built: 1995, neighborhood: 'Downtown', prop_type: 'Single Family', garage: 2, price: 465000 },
  { sqft: 1650, beds: 2, baths: 1.5, year_built: 1980, neighborhood: 'Suburbs', prop_type: 'Condo', garage: 1, price: 240000 },
  { sqft: 3100, beds: 4, baths: 3.0, year_built: 2010, neighborhood: 'Lakeside', prop_type: 'Single Family', garage: 3, price: 680000 },
  { sqft: 1420, beds: 2, baths: 1.0, year_built: 1975, neighborhood: 'Suburbs', prop_type: 'Townhouse', garage: 1, price: 210000 },
  { sqft: 2450, beds: 4, baths: 2.5, year_built: 2005, neighborhood: 'Uptown', prop_type: 'Single Family', garage: 2, price: 520000 },
  { sqft: 1850, beds: 3, baths: 2.0, year_built: 1990, neighborhood: 'Suburbs', prop_type: 'Single Family', garage: 2, price: 335000 },
  { sqft: 3400, beds: 5, baths: 3.5, year_built: 2018, neighborhood: 'Downtown', prop_type: 'Single Family', garage: 3, price: 780000 },
  { sqft: 1250, beds: 2, baths: 1.0, year_built: 1968, neighborhood: 'Suburbs', prop_type: 'Condo', garage: 0, price: 185000 },
  { sqft: 2750, beds: 4, baths: 3.0, year_built: 2002, neighborhood: 'Lakeside', prop_type: 'Single Family', garage: 2, price: 610000 },
  { sqft: 1950, beds: 3, baths: 2.0, year_built: 1998, neighborhood: 'Uptown', prop_type: 'Townhouse', garage: 2, price: 410000 },
  { sqft: 2200, beds: 3, baths: 2.5, year_built: 2006, neighborhood: 'Downtown', prop_type: 'Condo', garage: 1, price: 490000 },
  { sqft: 1720, beds: 3, baths: 1.5, year_built: 1985, neighborhood: 'Suburbs', prop_type: 'Single Family', garage: 1, price: 295000 },
  { sqft: 3650, beds: 5, baths: 4.0, year_built: 2021, neighborhood: 'Lakeside', prop_type: 'Single Family', garage: 3, price: 850000 },
  { sqft: 1550, beds: 2, baths: 2.0, year_built: 1992, neighborhood: 'Downtown', prop_type: 'Condo', garage: 1, price: 375000 },
  { sqft: 2900, beds: 4, baths: 3.0, year_built: 2012, neighborhood: 'Uptown', prop_type: 'Single Family', garage: 2, price: 595000 },
  { sqft: 2050, beds: 3, baths: 2.0, year_built: 2000, neighborhood: 'Suburbs', prop_type: 'Single Family', garage: 2, price: 360000 },
  { sqft: 1380, beds: 2, baths: 1.0, year_built: 1972, neighborhood: 'Suburbs', prop_type: 'Townhouse', garage: 1, price: 205000 },
  { sqft: 3250, beds: 4, baths: 3.5, year_built: 2015, neighborhood: 'Downtown', prop_type: 'Single Family', garage: 2, price: 730000 },
  { sqft: 2300, beds: 3, baths: 2.5, year_built: 2004, neighborhood: 'Lakeside', prop_type: 'Townhouse', garage: 2, price: 515000 },
  { sqft: 1680, beds: 3, baths: 1.5, year_built: 1988, neighborhood: 'Suburbs', prop_type: 'Single Family', garage: 1, price: 285000 },
];

const SAMPLE_CHURN_DATA: Record<string, any>[] = [
  { tenure_months: 2, monthly_charges: 70.35, total_charges: 140.70, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 3, churned: 1 },
  { tenure_months: 34, monthly_charges: 56.95, total_charges: 1936.30, contract: 'One year', internet: 'DSL', payment: 'Mailed check', support_calls: 0, churned: 0 },
  { tenure_months: 45, monthly_charges: 42.30, total_charges: 1903.50, contract: 'One year', internet: 'DSL', payment: 'Bank transfer', support_calls: 1, churned: 0 },
  { tenure_months: 8, monthly_charges: 89.10, total_charges: 712.80, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 4, churned: 1 },
  { tenure_months: 22, monthly_charges: 20.05, total_charges: 441.10, contract: 'Month-to-month', internet: 'No', payment: 'Mailed check', support_calls: 0, churned: 0 },
  { tenure_months: 10, monthly_charges: 95.50, total_charges: 955.00, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 5, churned: 1 },
  { tenure_months: 28, monthly_charges: 75.30, total_charges: 2108.40, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Credit card', support_calls: 2, churned: 0 },
  { tenure_months: 62, monthly_charges: 56.15, total_charges: 3481.30, contract: 'Two year', internet: 'DSL', payment: 'Bank transfer', support_calls: 0, churned: 0 },
  { tenure_months: 13, monthly_charges: 80.85, total_charges: 1051.05, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 3, churned: 1 },
  { tenure_months: 58, monthly_charges: 25.40, total_charges: 1473.20, contract: 'Two year', internet: 'No', payment: 'Credit card', support_calls: 0, churned: 0 },
  { tenure_months: 16, monthly_charges: 69.50, total_charges: 1112.00, contract: 'Month-to-month', internet: 'DSL', payment: 'Electronic check', support_calls: 2, churned: 0 },
  { tenure_months: 53, monthly_charges: 105.00, total_charges: 5565.00, contract: 'Two year', internet: 'Fiber optic', payment: 'Bank transfer', support_calls: 1, churned: 0 },
  { tenure_months: 1, monthly_charges: 74.40, total_charges: 74.40, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 3, churned: 1 },
  { tenure_months: 72, monthly_charges: 118.75, total_charges: 8550.00, contract: 'Two year', internet: 'Fiber optic', payment: 'Credit card', support_calls: 1, churned: 0 },
  { tenure_months: 5, monthly_charges: 85.20, total_charges: 426.00, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 4, churned: 1 },
  { tenure_months: 38, monthly_charges: 60.10, total_charges: 2283.80, contract: 'One year', internet: 'DSL', payment: 'Bank transfer', support_calls: 1, churned: 0 },
  { tenure_months: 65, monthly_charges: 102.50, total_charges: 6662.50, contract: 'Two year', internet: 'Fiber optic', payment: 'Credit card', support_calls: 0, churned: 0 },
  { tenure_months: 3, monthly_charges: 92.00, total_charges: 276.00, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 5, churned: 1 },
  { tenure_months: 40, monthly_charges: 24.80, total_charges: 992.00, contract: 'Two year', internet: 'No', payment: 'Mailed check', support_calls: 0, churned: 0 },
  { tenure_months: 15, monthly_charges: 84.60, total_charges: 1269.00, contract: 'Month-to-month', internet: 'Fiber optic', payment: 'Electronic check', support_calls: 2, churned: 1 },
];

function analyzeColumns(data: Record<string, any>[]): ColumnMeta[] {
  if (!data || data.length === 0) return [];
  const keys = Object.keys(data[0]);
  return keys.map((key) => {
    const values = data.map((d) => d[key]).filter((v) => v !== null && v !== undefined && v !== '');
    const isNum = values.every((v) => typeof v === 'number' || (!isNaN(Number(v)) && v !== ''));
    const uniqueVals = new Set(values);

    let stats: any = undefined;
    if (isNum && values.length > 0) {
      const numVals = values.map(Number).sort((a, b) => a - b);
      const sum = numVals.reduce((acc, val) => acc + val, 0);
      const mean = sum / numVals.length;
      const min = numVals[0];
      const max = numVals[numVals.length - 1];
      const p25 = numVals[Math.floor(numVals.length * 0.25)];
      const p50 = numVals[Math.floor(numVals.length * 0.5)];
      const p75 = numVals[Math.floor(numVals.length * 0.75)];
      const variance = numVals.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (numVals.length || 1);
      stats = {
        mean: Math.round(mean * 100) / 100,
        std: Math.round(Math.sqrt(variance) * 100) / 100,
        min,
        p25,
        p50,
        p75,
        max,
      };
    }

    const missing = data.length - values.length;
    return {
      name: key,
      dtype: isNum ? 'float64' : 'object',
      inferredType: isNum ? 'numerical' : 'categorical',
      missingCount: missing,
      missingPct: Math.round((missing / data.length) * 1000) / 10,
      uniqueCount: uniqueVals.size,
      sampleValues: Array.from(uniqueVals).slice(0, 4),
      stats,
    };
  });
}

const CustomKSTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload;
    return (
      <div className="bg-slate-900/95 border border-slate-700/80 rounded-xl p-3.5 shadow-2xl text-xs font-mono space-y-2.5 max-w-sm backdrop-blur-md">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-bold text-slate-200">Evaluation: {label}</span>
          </div>
          <span className="text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded font-mono">
            {dataPoint.batchId || 'Batch'}
          </span>
        </div>
        <div className="space-y-1.5 pt-0.5">
          {payload.map((item: any) => {
            const val = item.value;
            const isDrifted = typeof val === 'number' && val < 0.05;
            return (
              <div key={item.dataKey} className="flex items-center justify-between gap-4 py-0.5">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm" style={{ backgroundColor: item.color }} />
                  <span className="text-slate-300 font-semibold">{item.name}:</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="font-bold text-slate-100">{typeof val === 'number' ? val.toFixed(4) : val}</span>
                  {isDrifted ? (
                    <span className="text-[10px] px-1.5 py-0.2 bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded font-sans font-bold flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-pulse"></span> DRIFT (p&lt;0.05)
                    </span>
                  ) : (
                    <span className="text-[10px] px-1.5 py-0.2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded font-sans">
                      STABLE
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div className="text-[10px] text-slate-500 pt-1.5 border-t border-slate-800/80 flex items-center justify-between">
          <span>H₀: F_prod = F_ref (No Drift)</span>
          <span className="text-rose-400 font-bold">α Threshold = 0.05</span>
        </div>
      </div>
    );
  }
  return null;
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'datasets' | 'training' | 'experiments' | 'registry' | 'predictions' | 'monitoring' | 'apiDocs'>('overview');

  // Datasets State
  const [datasets, setDatasets] = useState<DatasetItem[]>([
    {
      id: 'ds_housing_prices',
      name: 'housing_prices.csv',
      rowCount: SAMPLE_HOUSING_DATA.length,
      colCount: Object.keys(SAMPLE_HOUSING_DATA[0]).length,
      taskType: 'regression',
      targetCol: 'price',
      columns: analyzeColumns(SAMPLE_HOUSING_DATA),
      data: SAMPLE_HOUSING_DATA,
      warnings: [],
    },
    {
      id: 'ds_customer_churn',
      name: 'customer_churn.csv',
      rowCount: SAMPLE_CHURN_DATA.length,
      colCount: Object.keys(SAMPLE_CHURN_DATA[0]).length,
      taskType: 'binary_classification',
      targetCol: 'churned',
      columns: analyzeColumns(SAMPLE_CHURN_DATA),
      data: SAMPLE_CHURN_DATA,
      warnings: [],
    },
  ]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('ds_housing_prices');

  // Model Training State
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>(['gradient_boosting', 'random_forest']);
  const [testSplit, setTestSplit] = useState<number>(0.2);
  const [cvFolds, setCvFolds] = useState<number>(3);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  // Models in Registry State
  const [models, setModels] = useState<ModelItem[]>([
    {
      id: 'mod_gb_housing_v1',
      name: 'Gradient Boosting Regressor',
      algorithm: 'gradient_boosting',
      taskType: 'regression',
      targetCol: 'price',
      version: 1,
      runId: 'run_b38290',
      metrics: { rmse: 31450.2, mae: 24100.5, r2: 0.941 },
      cvMean: 0.925,
      cvScores: [0.932, 0.918, 0.925],
      durationSec: 1.42,
      isActive: true,
      status: 'validated',
      createdAt: '2026-10-12T02:15:00Z',
      activatedAt: '2026-10-12T02:16:00Z',
      features: ['sqft', 'beds', 'baths', 'year_built', 'neighborhood', 'prop_type', 'garage'],
      referenceDataSummary: {
        sqft: { mean: 2210, std: 620, median: 2100 },
        beds: { mean: 3.2, std: 0.9, median: 3.0 },
        baths: { mean: 2.3, std: 0.8, median: 2.0 },
        year_built: { mean: 1996, std: 14.5, median: 1997 },
        garage: { mean: 1.8, std: 0.7, median: 2.0 },
      },
    },
    {
      id: 'mod_rf_housing_v1',
      name: 'Random Forest Regressor',
      algorithm: 'random_forest',
      taskType: 'regression',
      targetCol: 'price',
      version: 1,
      runId: 'run_a17103',
      metrics: { rmse: 35800.0, mae: 27900.2, r2: 0.912 },
      cvMean: 0.894,
      cvScores: [0.901, 0.885, 0.896],
      durationSec: 1.15,
      isActive: false,
      status: 'validated',
      createdAt: '2026-10-12T02:15:00Z',
      features: ['sqft', 'beds', 'baths', 'year_built', 'neighborhood', 'prop_type', 'garage'],
      referenceDataSummary: {
        sqft: { mean: 2210, std: 620, median: 2100 },
        beds: { mean: 3.2, std: 0.9, median: 3.0 },
        baths: { mean: 2.3, std: 0.8, median: 2.0 },
        year_built: { mean: 1996, std: 14.5, median: 1997 },
        garage: { mean: 1.8, std: 0.7, median: 2.0 },
      },
    },
  ]);

  // Prediction Form State
  const activeModel = useMemo(() => models.find((m) => m.isActive) || models[0], [models]);
  const [predictionInputs, setPredictionInputs] = useState<Record<string, any>>({
    sqft: 2400,
    beds: 3,
    baths: 2.5,
    year_built: 2005,
    neighborhood: 'Downtown',
    prop_type: 'Single Family',
    garage: 2,
    tenure_months: 6,
    monthly_charges: 85.0,
    total_charges: 510.0,
    contract: 'Month-to-month',
    internet: 'Fiber optic',
    payment: 'Electronic check',
    support_calls: 3,
  });
  const [latestPrediction, setLatestPrediction] = useState<PredictionLog | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);

  // Monitoring & Telemetry State
  const [predictionLogs, setPredictionLogs] = useState<PredictionLog[]>([
    {
      id: 'req_101',
      modelId: 'mod_gb_housing_v1',
      modelVersion: 1,
      features: { sqft: 2200, beds: 3, baths: 2.0, year_built: 1998, neighborhood: 'Downtown', prop_type: 'Single Family', garage: 2 },
      prediction: 472500,
      latencyMs: 14.2,
      timestamp: '2026-10-12T03:10:00Z',
    },
    {
      id: 'req_102',
      modelId: 'mod_gb_housing_v1',
      modelVersion: 1,
      features: { sqft: 1800, beds: 2, baths: 1.5, year_built: 1985, neighborhood: 'Suburbs', prop_type: 'Condo', garage: 1 },
      prediction: 285000,
      latencyMs: 12.8,
      timestamp: '2026-10-12T03:12:00Z',
    },
    {
      id: 'req_103',
      modelId: 'mod_gb_housing_v1',
      modelVersion: 1,
      features: { sqft: 3100, beds: 4, baths: 3.0, year_built: 2015, neighborhood: 'Lakeside', prop_type: 'Single Family', garage: 3 },
      prediction: 698000,
      latencyMs: 18.5,
      timestamp: '2026-10-12T03:15:00Z',
    },
  ]);

  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: 'alt_902',
      modelId: 'mod_gb_housing_v1',
      type: 'DATA_DRIFT',
      severity: 'WARNING',
      message: 'Feature drift detected in square_feet (KS p-value: 0.021 < 0.05)',
      details: 'Recent production observations show shifted square footage distribution toward > 3500 sqft.',
      status: 'ACTIVE',
      createdAt: '2026-10-12T03:00:00Z',
    },
  ]);

  // Historical Kolmogorov-Smirnov p-value tracking state
  const [housingKSHistory, setHousingKSHistory] = useState<HistoricalKSEntry[]>(INITIAL_HOUSING_KS_HISTORY);
  const [churnKSHistory, setChurnKSHistory] = useState<HistoricalKSEntry[]>(INITIAL_CHURN_KS_HISTORY);
  const [disabledChartFeatures, setDisabledChartFeatures] = useState<Record<string, boolean>>({});

  // Drift Report JSON Export & Inspection Modal State
  const [jsonModalOpen, setJsonModalOpen] = useState<boolean>(false);
  const [jsonExportContent, setJsonExportContent] = useState<string>('');
  const [exportToastMessage, setExportToastMessage] = useState<string | null>(null);
  const [copiedJson, setCopiedJson] = useState<boolean>(false);

  // Active KS Historical Data based on active model's task type
  const activeKSHistory = useMemo(() => {
    return activeModel.taskType === 'regression' ? housingKSHistory : churnKSHistory;
  }, [activeModel.taskType, housingKSHistory, churnKSHistory]);

  // Active Numerical Features for the current model
  const activeNumericFeatures = useMemo(() => {
    if (activeModel.taskType === 'regression') {
      return ['sqft', 'beds', 'baths', 'year_built', 'garage'];
    } else {
      return ['tenure_months', 'monthly_charges', 'total_charges', 'support_calls'];
    }
  }, [activeModel.taskType]);

  // Features to plot on the Line Chart (ignoring toggled off ones)
  const visibleChartFeatures = useMemo(() => {
    return activeNumericFeatures.filter((f) => !disabledChartFeatures[f]);
  }, [activeNumericFeatures, disabledChartFeatures]);

  const toggleFeatureVisibility = (feature: string) => {
    setDisabledChartFeatures((prev) => ({
      ...prev,
      [feature]: !prev[feature],
    }));
  };

  const handleRunManualKSEval = () => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const batchId = `eval_${now.getMinutes()}${now.getSeconds()}`;
    const isReg = activeModel.taskType === 'regression';

    if (isReg) {
      const newPoint: HistoricalKSEntry = {
        time: timeStr,
        timestamp: now.toISOString(),
        batchId,
        sqft: Math.round((Math.random() * 0.4 + 0.35) * 1000) / 1000,
        beds: Math.round((Math.random() * 0.3 + 0.6) * 1000) / 1000,
        baths: Math.round((Math.random() * 0.3 + 0.65) * 1000) / 1000,
        year_built: Math.round((Math.random() * 0.3 + 0.55) * 1000) / 1000,
        garage: Math.round((Math.random() * 0.3 + 0.65) * 1000) / 1000,
      };
      setHousingKSHistory((prev) => [...prev, newPoint]);
    } else {
      const newPoint: HistoricalKSEntry = {
        time: timeStr,
        timestamp: now.toISOString(),
        batchId,
        tenure_months: Math.round((Math.random() * 0.3 + 0.6) * 1000) / 1000,
        monthly_charges: Math.round((Math.random() * 0.3 + 0.55) * 1000) / 1000,
        total_charges: Math.round((Math.random() * 0.3 + 0.6) * 1000) / 1000,
        support_calls: Math.round((Math.random() * 0.3 + 0.65) * 1000) / 1000,
      };
      setChurnKSHistory((prev) => [...prev, newPoint]);
    }
  };

  const handleResetKSHistory = () => {
    if (activeModel.taskType === 'regression') {
      setHousingKSHistory(INITIAL_HOUSING_KS_HISTORY);
    } else {
      setChurnKSHistory(INITIAL_CHURN_KS_HISTORY);
    }
    setDisabledChartFeatures({});
  };

  const currentDataset = useMemo(() => {
    return datasets.find((d) => d.id === selectedDatasetId) || datasets[0];
  }, [datasets, selectedDatasetId]);

  // Telemetry Calculations
  const telemetry = useMemo(() => {
    const total = predictionLogs.length;
    if (total === 0) return { total: 0, avgLatency: 0, p50: 0, p95: 0, p99: 0 };
    const lats = predictionLogs.map((p) => p.latencyMs).sort((a, b) => a - b);
    const sum = lats.reduce((acc, v) => acc + v, 0);
    return {
      total,
      avgLatency: Math.round((sum / total) * 10) / 10,
      p50: Math.round(lats[Math.floor(total * 0.5)] * 10) / 10,
      p95: Math.round(lats[Math.floor(total * 0.95)] * 10) / 10,
      p99: Math.round(lats[Math.floor(total * 0.99)] * 10) / 10,
    };
  }, [predictionLogs]);

  // Execute Simulated Training Pipeline
  const runTraining = () => {
    setIsTraining(true);
    setTrainingLogs([
      `[1/6] Ingesting dataset '${currentDataset.name}' (${currentDataset.rowCount} rows, ${currentDataset.colCount} cols)...`,
    ]);

    setTimeout(() => {
      setTrainingLogs((prev) => [
        ...prev,
        `[2/6] Validating schemas: 0 duplicate rows, 0 target leakage columns detected. Target: '${currentDataset.targetCol}' (${currentDataset.taskType}).`,
      ]);
    }, 400);

    setTimeout(() => {
      setTrainingLogs((prev) => [
        ...prev,
        `[3/6] Preprocessing pipeline: Median imputer (numerical) + Most Frequent (categorical) + OneHotEncoder(handle_unknown='ignore') fitted strictly on ${100 - testSplit * 100}% training fold.`,
      ]);
    }, 800);

    setTimeout(() => {
      setTrainingLogs((prev) => [
        ...prev,
        `[4/6] Running ${cvFolds}-Fold Stratified Cross-Validation across: ${selectedAlgorithms.join(', ')}...`,
      ]);
    }, 1200);

    setTimeout(() => {
      setTrainingLogs((prev) => [
        ...prev,
        `[5/6] Calculating evaluation metrics on held-out ${testSplit * 100}% test set...`,
      ]);
    }, 1600);

    setTimeout(() => {
      // Create new trained models
      const newModels: ModelItem[] = selectedAlgorithms.map((algo) => {
        const isReg = currentDataset.taskType === 'regression';
        const version = models.filter((m) => m.algorithm === algo).length + 1;
        const metrics: Record<string, number> = isReg
          ? {
              rmse: algo === 'gradient_boosting' ? 30200 : algo === 'random_forest' ? 34100 : 42500,
              mae: algo === 'gradient_boosting' ? 22800 : algo === 'random_forest' ? 26200 : 33000,
              r2: algo === 'gradient_boosting' ? 0.952 : algo === 'random_forest' ? 0.928 : 0.865,
            }
          : {
              accuracy: algo === 'gradient_boosting' ? 0.885 : algo === 'random_forest' ? 0.852 : 0.811,
              f1: algo === 'gradient_boosting' ? 0.872 : algo === 'random_forest' ? 0.839 : 0.795,
              precision: algo === 'gradient_boosting' ? 0.865 : algo === 'random_forest' ? 0.842 : 0.801,
              recall: algo === 'gradient_boosting' ? 0.880 : algo === 'random_forest' ? 0.835 : 0.790,
              roc_auc: algo === 'gradient_boosting' ? 0.912 : algo === 'random_forest' ? 0.890 : 0.845,
            };

        return {
          id: `mod_${algo}_${Date.now().toString(36)}`,
          name: algo.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
          algorithm: algo,
          taskType: currentDataset.taskType,
          targetCol: currentDataset.targetCol,
          version,
          runId: `run_${Math.random().toString(36).substring(2, 8)}`,
          metrics,
          cvMean: isReg ? 0.93 : 0.86,
          cvScores: [0.92, 0.94, 0.93],
          durationSec: Math.round((Math.random() * 0.8 + 0.6) * 100) / 100,
          isActive: false,
          status: 'validated',
          createdAt: new Date().toISOString(),
          features: currentDataset.columns.filter((c) => c.name !== currentDataset.targetCol).map((c) => c.name),
          referenceDataSummary: {
            sqft: { mean: 2210, std: 620, median: 2100 },
            monthly_charges: { mean: 68.5, std: 24.2, median: 70.0 },
          },
        };
      });

      // Best model auto-promotion
      const best = newModels[0];
      best.isActive = true;
      best.activatedAt = new Date().toISOString();

      setModels((prev) => [
        best,
        ...newModels.slice(1),
        ...prev.map((m) => ({ ...m, isActive: false })),
      ]);

      setTrainingLogs((prev) => [
        ...prev,
        `[6/6] [SUCCESS] All ${selectedAlgorithms.length} models trained & logged to MLflow! Candidate '${best.name}' auto-promoted to active serving.`,
      ]);
      setIsTraining(false);
    }, 2000);
  };

  // Run Real-Time Prediction
  const handlePredict = () => {
    setIsPredicting(true);
    setTimeout(() => {
      const isReg = activeModel.taskType === 'regression';
      let predVal: any;
      let probs: Record<string, number> | undefined = undefined;

      if (isReg) {
        // Linear synthetic regression formula
        const sqft = Number(predictionInputs.sqft) || 2000;
        const beds = Number(predictionInputs.beds) || 3;
        const baths = Number(predictionInputs.baths) || 2;
        const year = Number(predictionInputs.year_built) || 2000;
        const garage = Number(predictionInputs.garage) || 1;
        const neighborhoodMulti = predictionInputs.neighborhood === 'Lakeside' ? 1.4 : (predictionInputs.neighborhood === 'Downtown' ? 1.3 : 1.0);
        predVal = Math.round((120000 + sqft * 135 + beds * 14000 + baths * 18000 + (year - 1960) * 1100 + garage * 12500) * neighborhoodMulti);
      } else {
        // Binary classification formula
        const tenure = Number(predictionInputs.tenure_months) || 12;
        const charges = Number(predictionInputs.monthly_charges) || 70;
        const calls = Number(predictionInputs.support_calls) || 1;
        const z = -1.5 - 0.04 * tenure + 0.02 * charges + 0.4 * calls + (predictionInputs.contract === 'Month-to-month' ? 1.2 : -0.8);
        const p = 1 / (1 + Math.exp(-z));
        predVal = p > 0.5 ? 1 : 0;
        probs = {
          'No Churn (0)': Math.round((1 - p) * 1000) / 1000,
          'Churn (1)': Math.round(p * 1000) / 1000,
        };
      }

      const latency = Math.round((Math.random() * 8 + 9) * 10) / 10;
      const newLog: PredictionLog = {
        id: `req_${Date.now().toString(36)}`,
        modelId: activeModel.id,
        modelVersion: activeModel.version,
        features: { ...predictionInputs },
        prediction: predVal,
        probabilities: probs,
        latencyMs: latency,
        timestamp: new Date().toISOString(),
      };

      setPredictionLogs((prev) => [newLog, ...prev]);
      setLatestPrediction(newLog);
      setIsPredicting(false);
    }, 280);
  };

  // Simulate Shifted Traffic for Drift Demonstration
  const handleSimulateDriftTraffic = () => {
    const isReg = activeModel.taskType === 'regression';
    const batch: PredictionLog[] = [];
    for (let i = 0; i < 25; i++) {
      const latency = Math.round((Math.random() * 12 + 10) * 10) / 10;
      if (isReg) {
        // Shifted sqft: 3600 - 4500 sqft instead of baseline 2100
        const shiftedSqft = Math.round(3600 + Math.random() * 900);
        batch.push({
          id: `req_drift_${i}_${Date.now().toString(36)}`,
          modelId: activeModel.id,
          modelVersion: activeModel.version,
          features: {
            sqft: shiftedSqft,
            beds: 4,
            baths: 3.5,
            year_built: 2018,
            neighborhood: 'Lakeside',
            prop_type: 'Single Family',
            garage: 3,
          },
          prediction: 790000 + shiftedSqft * 100,
          latencyMs: latency,
          timestamp: new Date().toISOString(),
        });
      } else {
        // Shifted tenure and support calls
        batch.push({
          id: `req_drift_${i}_${Date.now().toString(36)}`,
          modelId: activeModel.id,
          modelVersion: activeModel.version,
          features: {
            tenure_months: 2,
            monthly_charges: 110 + Math.random() * 20,
            total_charges: 250,
            contract: 'Month-to-month',
            internet: 'Fiber optic',
            payment: 'Electronic check',
            support_calls: 5 + Math.floor(Math.random() * 3),
          },
          prediction: 1,
          probabilities: { 'No Churn (0)': 0.12, 'Churn (1)': 0.88 },
          latencyMs: latency,
          timestamp: new Date().toISOString(),
        });
      }
    }

    setPredictionLogs((prev) => [...batch, ...prev]);

    // Append new historical point to KS drift history
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const batchId = `eval_${now.getMinutes()}${now.getSeconds()}`;

    if (isReg) {
      const newPoint: HistoricalKSEntry = {
        time: timeStr,
        timestamp: now.toISOString(),
        batchId,
        sqft: 0.0031, // drifted!
        beds: 0.628,
        baths: 0.694,
        year_built: 0.541,
        garage: 0.725,
      };
      setHousingKSHistory((prev) => [...prev, newPoint]);
    } else {
      const newPoint: HistoricalKSEntry = {
        time: timeStr,
        timestamp: now.toISOString(),
        batchId,
        tenure_months: 0.0001,   // drifted!
        monthly_charges: 0.0001, // drifted!
        total_charges: 0.0084,   // drifted!
        support_calls: 0.0001,   // drifted!
      };
      setChurnKSHistory((prev) => [...prev, newPoint]);
    }

    // Add alert
    const newAlert: Alert = {
      id: `alt_${Date.now().toString(36)}`,
      modelId: activeModel.id,
      type: 'DATA_DRIFT',
      severity: 'WARNING',
      message: isReg
        ? 'Kolmogorov-Smirnov test detected significant drift on feature: square_feet (p-value: 0.003 < 0.05)'
        : 'Statistical shift detected on monthly_charges (p-value: 0.008 < 0.05) & support_calls',
      details: '25 production inference payloads evaluated against baseline reference training set.',
      status: 'ACTIVE',
      createdAt: new Date().toISOString(),
    };
    setAlerts((prev) => [newAlert, ...prev]);
  };

  // Construct Comprehensive Drift Report Payload with KS Statistics and Model Metadata
  const generateDriftReportPayload = () => {
    const latestKS = activeKSHistory[activeKSHistory.length - 1] || {};

    const featuresReport = activeNumericFeatures.map((feat) => {
      const pVal = typeof latestKS[feat] === 'number' ? latestKS[feat] : 0.85;
      const refStats =
        (activeModel.referenceDataSummary && activeModel.referenceDataSummary[feat]) ||
        currentDataset.columns.find((c) => c.name === feat)?.stats || {
          mean: 100,
          std: 15,
          median: 95,
        };
      const isDrifted = pVal < 0.05;
      const shiftFactor = isDrifted ? 1.45 : 1.02;
      return {
        feature_name: feat,
        feature_type: 'numerical',
        test_method: 'Two-sample Kolmogorov-Smirnov test',
        ks_statistic: isDrifted ? 0.642 : 0.085,
        p_value: pVal,
        significance_threshold: 0.05,
        drift_detected: isDrifted,
        reference_baseline: {
          mean: refStats.mean,
          std: refStats.std,
          median: (refStats as any).median ?? (refStats as any).p50 ?? refStats.mean,
        },
        production_current: {
          mean: Math.round(refStats.mean * shiftFactor * 100) / 100,
          std: Math.round(refStats.std * (isDrifted ? 1.3 : 1.0) * 100) / 100,
          sample_size: Math.max(predictionLogs.length, 25),
        },
      };
    });

    const driftedCount = featuresReport.filter((f) => f.drift_detected).length;

    return {
      report_title: 'MLGuard Production Data Drift & Kolmogorov-Smirnov Report',
      exported_at: new Date().toISOString(),
      platform: 'MLGuard MLOps Platform v1.0',
      active_model: {
        model_id: activeModel.id,
        name: activeModel.name,
        algorithm: activeModel.algorithm,
        version: activeModel.version,
        task_type: activeModel.taskType,
        target_column: activeModel.targetCol,
        deployment_status: activeModel.isActive ? 'ACTIVE_SERVING' : 'INACTIVE',
        activated_at: activeModel.activatedAt || activeModel.createdAt,
        holdout_metrics: activeModel.metrics,
        features_contract: activeModel.features,
      },
      monitoring_telemetry: {
        total_inferences_logged: predictionLogs.length,
        avg_latency_ms: telemetry.avgLatency,
        p50_latency_ms: telemetry.p50,
        p95_latency_ms: telemetry.p95,
        p99_latency_ms: telemetry.p99,
        significance_threshold_alpha: 0.05,
        overall_drift_detected: driftedCount > 0,
        drifted_features_count: driftedCount,
        analyzed_features_count: activeNumericFeatures.length,
      },
      kolmogorov_smirnov_features: featuresReport,
      historical_p_values_series: activeKSHistory,
      active_alerts: alerts.filter((a) => a.modelId === activeModel.id),
    };
  };

  // Export Current Drift Report as Downloadable JSON File
  const handleExportDriftReport = () => {
    const payload = generateDriftReportPayload();
    const jsonStr = JSON.stringify(payload, null, 2);
    const jsonBlob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(jsonBlob);
    const filename = `drift_report_${activeModel.algorithm}_v${activeModel.version}_${new Date().toISOString().slice(0, 10)}.json`;
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    setExportToastMessage(`Exported: ${filename} (${(jsonBlob.size / 1024).toFixed(1)} KB)`);
    setTimeout(() => setExportToastMessage(null), 4000);
  };

  // Open Preview Modal of Current Drift Report JSON
  const handlePreviewDriftReport = () => {
    const payload = generateDriftReportPayload();
    setJsonExportContent(JSON.stringify(payload, null, 2));
    setCopiedJson(false);
    setJsonModalOpen(true);
  };

  // Copy JSON content to clipboard
  const handleCopyJson = () => {
    navigator.clipboard.writeText(jsonExportContent);
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2500);
  };

  // Run full demo workflow in 1 click
  const handleRunFullDemoWorkflow = () => {
    runTraining();
    setTimeout(() => {
      handleSimulateDriftTraffic();
    }, 2200);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Banner / System Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
                MLGuard
              </span>
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                v1.0 Production
              </span>
            </div>
            <p className="text-xs text-slate-400">Automated Tabular ML Training, Serving & Drift Monitoring</p>
          </div>
        </div>

        {/* Live Service Indicators */}
        <div className="hidden lg:flex items-center gap-4 text-xs font-mono text-slate-300">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>FastAPI:8000</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span>SQLite DB</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60">
            <span className="w-2 h-2 rounded-full bg-indigo-400"></span>
            <span>MLflow Tracking</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3">
          <button
            onClick={handleRunFullDemoWorkflow}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white shadow-md shadow-indigo-500/25 transition cursor-pointer"
          >
            <Sparkles className="w-4 h-4" />
            <span>Run 1-Click Demo Pipeline</span>
          </button>
        </div>
      </header>

      {/* Main Layout Container */}
      <div className="flex-1 flex flex-col md:flex-row">
        {/* Sidebar Navigation */}
        <aside className="w-full md:w-64 border-r border-slate-800/80 bg-slate-900/40 p-4 shrink-0 flex flex-col justify-between">
          <nav className="space-y-1">
            <button
              onClick={() => setActiveTab('overview')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'overview' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <Activity className="w-4 h-4 text-indigo-400" />
              <span>System Overview</span>
            </button>

            <button
              onClick={() => setActiveTab('datasets')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'datasets' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <Database className="w-4 h-4 text-cyan-400" />
              <span>Dataset Explorer</span>
            </button>

            <button
              onClick={() => setActiveTab('training')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'training' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <Cpu className="w-4 h-4 text-amber-400" />
              <span>Train Models</span>
            </button>

            <button
              onClick={() => setActiveTab('experiments')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'experiments' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <BarChart3 className="w-4 h-4 text-violet-400" />
              <span>Experiments & MLflow</span>
            </button>

            <button
              onClick={() => setActiveTab('registry')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'registry' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <Layers className="w-4 h-4 text-pink-400" />
              <span>Model Registry</span>
            </button>

            <button
              onClick={() => setActiveTab('predictions')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'predictions' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <Zap className="w-4 h-4 text-emerald-400" />
              <span>Prediction Playground</span>
            </button>

            <button
              onClick={() => setActiveTab('monitoring')}
              className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'monitoring' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center gap-3">
                <Gauge className="w-4 h-4 text-rose-400" />
                <span>Monitoring & Drift</span>
              </div>
              {alerts.filter((a) => a.status === 'ACTIVE').length > 0 && (
                <span className="w-2 h-2 rounded-full bg-rose-500"></span>
              )}
            </button>

            <button
              onClick={() => setActiveTab('apiDocs')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${
                activeTab === 'apiDocs' ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }`}
            >
              <FileCode className="w-4 h-4 text-blue-400" />
              <span>Architecture & API</span>
            </button>
          </nav>

          {/* Active Serving Mini Card */}
          <div className="mt-8 p-3 rounded-xl bg-slate-900 border border-slate-800/90 text-xs">
            <div className="flex items-center justify-between text-slate-400 mb-1.5">
              <span>Active Model</span>
              <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span> Live
              </span>
            </div>
            <p className="font-semibold text-slate-200 truncate">{activeModel.name}</p>
            <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1 font-mono">
              <span>v{activeModel.version}</span>
              <span className="text-indigo-400">{activeModel.taskType}</span>
            </div>
          </div>
        </aside>

        {/* Content View */}
        <main className="flex-1 p-6 md:p-8 overflow-y-auto max-w-7xl mx-auto w-full">
          {/* TAB 1: OVERVIEW */}
          {activeTab === 'overview' && (
            <div className="space-y-8">
              {/* Header */}
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">System Status & MLOps Pipeline</h1>
                <p className="text-slate-400 text-sm mt-1">
                  Automated tabular ML lifecycle with immutable model registry, real-time inference, and KS/Chi-square drift detection.
                </p>
              </div>

              {/* KPI Metrics Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                    <span>Registered Models</span>
                    <Layers className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">{models.length}</div>
                  <div className="text-xs text-indigo-400 mt-1 flex items-center gap-1">
                    <span>Active: {activeModel.name} (v{activeModel.version})</span>
                  </div>
                </div>

                <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                    <span>Total Inferences</span>
                    <Zap className="w-4 h-4 text-emerald-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">{telemetry.total}</div>
                  <div className="text-xs text-emerald-400 mt-1">FastAPI served requests</div>
                </div>

                <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                    <span>p95 Latency</span>
                    <Gauge className="w-4 h-4 text-amber-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">{telemetry.p95} ms</div>
                  <div className="text-xs text-slate-400 mt-1">Threshold: 250.0 ms</div>
                </div>

                <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                    <span>Active Alerts</span>
                    <AlertTriangle className="w-4 h-4 text-rose-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">
                    {alerts.filter((a) => a.status === 'ACTIVE').length}
                  </div>
                  <div className="text-xs text-rose-400 mt-1">
                    {alerts.filter((a) => a.status === 'ACTIVE').length > 0 ? 'Drift detected' : 'Normal state'}
                  </div>
                </div>
              </div>

              {/* End-to-End Architecture Flow */}
              <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-400" />
                    <span>MLGuard Production Architecture Flow</span>
                  </h2>
                  <span className="text-xs text-slate-400">Deterministic & Leakage-Free</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-xs">
                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                    <div className="text-indigo-400 font-mono font-bold mb-1">01. INGESTION</div>
                    <div className="font-semibold text-slate-200">CSV Validation</div>
                    <div className="text-slate-400 text-[11px] mt-1">Checks row/col limits, constant cols, ID columns & leakage risk.</div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                    <div className="text-cyan-400 font-mono font-bold mb-1">02. PIPELINE</div>
                    <div className="font-semibold text-slate-200">ColumnTransformer</div>
                    <div className="text-slate-400 text-[11px] mt-1">Fitted strictly on train fold. Handles median imputation & OneHot.</div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                    <div className="text-amber-400 font-mono font-bold mb-1">03. TRAINING</div>
                    <div className="font-semibold text-slate-200">K-Fold & MLflow</div>
                    <div className="text-slate-400 text-[11px] mt-1">RF & Gradient Boosting. Logs parameters, metrics & artifacts.</div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                    <div className="text-pink-400 font-mono font-bold mb-1">04. SERVING</div>
                    <div className="font-semibold text-slate-200">Model Registry</div>
                    <div className="text-slate-400 text-[11px] mt-1">Versioned immutable models with 1-click active serving promotion.</div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                    <div className="text-emerald-400 font-mono font-bold mb-1">05. MONITORING</div>
                    <div className="font-semibold text-slate-200">KS Drift & Alerts</div>
                    <div className="text-slate-400 text-[11px] mt-1">Two-sample Kolmogorov-Smirnov test against baseline training set.</div>
                  </div>
                </div>
              </div>

              {/* Quick Launch Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-6 rounded-2xl bg-gradient-to-br from-indigo-950/40 to-slate-900 border border-indigo-900/40">
                  <h3 className="text-base font-semibold text-white mb-2 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-indigo-400" />
                    <span>Instant Live Inference</span>
                  </h3>
                  <p className="text-sm text-slate-400 mb-4">
                    Send test requests to currently active model <strong className="text-slate-200">{activeModel.name}</strong> and inspect real-time latency breakdown.
                  </p>
                  <button
                    onClick={() => setActiveTab('predictions')}
                    className="inline-flex items-center gap-2 text-xs font-semibold text-indigo-400 hover:text-indigo-300 cursor-pointer"
                  >
                    <span>Open Prediction Playground</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>

                <div className="p-6 rounded-2xl bg-gradient-to-br from-rose-950/40 to-slate-900 border border-rose-900/40">
                  <h3 className="text-base font-semibold text-white mb-2 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-rose-400" />
                    <span>Data Drift & Telemetry</span>
                  </h3>
                  <p className="text-sm text-slate-400 mb-4">
                    Simulate shifted customer or property features to test Kolmogorov-Smirnov hypothesis tests and verify real-time alert generation.
                  </p>
                  <button
                    onClick={() => setActiveTab('monitoring')}
                    className="inline-flex items-center gap-2 text-xs font-semibold text-rose-400 hover:text-rose-300 cursor-pointer"
                  >
                    <span>View Drift Detection Suite</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: DATASET EXPLORER */}
          {activeTab === 'datasets' && (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold text-white tracking-tight">Dataset Explorer & Validation</h1>
                  <p className="text-slate-400 text-sm mt-1">
                    Upload tabular CSV files, inspect schema types, detect data leakage, and check missingness.
                  </p>
                </div>

                {/* Pre-loaded dataset switchers */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Load Sample:</span>
                  <button
                    onClick={() => setSelectedDatasetId('ds_housing_prices')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition cursor-pointer ${
                      selectedDatasetId === 'ds_housing_prices'
                        ? 'bg-indigo-600 text-white border-indigo-500'
                        : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800'
                    }`}
                  >
                    Housing Prices (Regression)
                  </button>
                  <button
                    onClick={() => setSelectedDatasetId('ds_customer_churn')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition cursor-pointer ${
                      selectedDatasetId === 'ds_customer_churn'
                        ? 'bg-indigo-600 text-white border-indigo-500'
                        : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800'
                    }`}
                  >
                    Customer Churn (Classification)
                  </button>
                </div>
              </div>

              {/* Dataset Metadata Bar */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-xs text-slate-400">Filename</div>
                  <div className="text-sm font-semibold text-slate-200 mt-1 font-mono">{currentDataset.name}</div>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-xs text-slate-400">Dimensions</div>
                  <div className="text-sm font-semibold text-slate-200 mt-1">
                    {currentDataset.rowCount} rows × {currentDataset.colCount} columns
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-xs text-slate-400">Target Column</div>
                  <div className="text-sm font-semibold text-indigo-400 mt-1 font-mono">{currentDataset.targetCol}</div>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-xs text-slate-400">Suggested Task</div>
                  <div className="text-sm font-semibold text-emerald-400 mt-1 uppercase text-xs">{currentDataset.taskType}</div>
                </div>
              </div>

              {/* Schema Table */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
                <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                  <h3 className="font-semibold text-sm text-slate-200">Column Schema & Statistical Profiling</h3>
                  <span className="text-xs text-slate-400">{currentDataset.columns.length} Total Features</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900 text-slate-400 font-mono border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-3">Feature Name</th>
                        <th className="px-4 py-3">Data Type</th>
                        <th className="px-4 py-3">Inferred</th>
                        <th className="px-4 py-3">Missing</th>
                        <th className="px-4 py-3">Unique</th>
                        <th className="px-4 py-3">Mean ± Std</th>
                        <th className="px-4 py-3">Range [Min - Max]</th>
                        <th className="px-4 py-3">Sample Values</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {currentDataset.columns.map((col) => (
                        <tr key={col.name} className="hover:bg-slate-800/40">
                          <td className="px-4 py-3 font-semibold text-slate-200 font-mono">
                            {col.name}
                            {col.name === currentDataset.targetCol && (
                              <span className="ml-2 text-[10px] bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded">
                                TARGET
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3 font-mono text-slate-400">{col.dtype}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                              col.inferredType === 'numerical' ? 'bg-cyan-500/10 text-cyan-400' : 'bg-pink-500/10 text-pink-400'
                            }`}>
                              {col.inferredType}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {col.missingPct > 0 ? (
                              <span className="text-amber-400">{col.missingPct}%</span>
                            ) : (
                              <span className="text-emerald-400">0%</span>
                            )}
                          </td>
                          <td className="px-4 py-3">{col.uniqueCount}</td>
                          <td className="px-4 py-3 font-mono">
                            {col.stats ? `${col.stats.mean} ± ${col.stats.std}` : '—'}
                          </td>
                          <td className="px-4 py-3 font-mono">
                            {col.stats ? `[${col.stats.min}, ${col.stats.max}]` : '—'}
                          </td>
                          <td className="px-4 py-3 text-slate-400 truncate max-w-xs">
                            {col.sampleValues.join(', ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Data Preview */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
                <div className="p-4 border-b border-slate-800">
                  <h3 className="font-semibold text-sm text-slate-200">Raw Data Preview (First 8 Rows)</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300 font-mono">
                    <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                      <tr>
                        {currentDataset.columns.map((c) => (
                          <th key={c.name} className="px-4 py-2.5">{c.name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {currentDataset.data.slice(0, 8).map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/40">
                          {currentDataset.columns.map((c) => (
                            <td key={c.name} className="px-4 py-2 text-slate-300">{row[c.name]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: TRAIN MODELS */}
          {activeTab === 'training' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">Machine Learning Training Engine</h1>
                <p className="text-slate-400 text-sm mt-1">
                  Configure training runs, cross-validation folds, feature transformers, and compare holdout scores.
                </p>
              </div>

              {/* Configuration Form */}
              <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-2">Selected Dataset</label>
                    <select
                      value={selectedDatasetId}
                      onChange={(e) => setSelectedDatasetId(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                    >
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>{d.name} ({d.taskType})</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-2">Target Variable</label>
                    <div className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-indigo-400 font-mono">
                      {currentDataset.targetCol}
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-2">Prediction Task</label>
                    <div className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-emerald-400 uppercase font-mono">
                      {currentDataset.taskType}
                    </div>
                  </div>
                </div>

                {/* Algorithm Selection */}
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-2">Algorithms to Train & Compare</label>
                  <div className="flex flex-wrap gap-3">
                    {currentDataset.taskType === 'regression' ? (
                      <>
                        {['gradient_boosting', 'random_forest', 'linear_regression'].map((algo) => (
                          <label
                            key={algo}
                            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg border text-xs font-medium cursor-pointer transition ${
                              selectedAlgorithms.includes(algo)
                                ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300'
                                : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                            }`}
                          >
                            <input
                              type="checkbox"
                              className="hidden"
                              checked={selectedAlgorithms.includes(algo)}
                              onChange={() => {
                                setSelectedAlgorithms((prev) =>
                                  prev.includes(algo) ? prev.filter((a) => a !== algo) : [...prev, algo]
                                );
                              }}
                            />
                            <span>{algo.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}</span>
                          </label>
                        ))}
                      </>
                    ) : (
                      <>
                        {['gradient_boosting', 'random_forest', 'logistic_regression'].map((algo) => (
                          <label
                            key={algo}
                            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg border text-xs font-medium cursor-pointer transition ${
                              selectedAlgorithms.includes(algo)
                                ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300'
                                : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                            }`}
                          >
                            <input
                              type="checkbox"
                              className="hidden"
                              checked={selectedAlgorithms.includes(algo)}
                              onChange={() => {
                                setSelectedAlgorithms((prev) =>
                                  prev.includes(algo) ? prev.filter((a) => a !== algo) : [...prev, algo]
                                );
                              }}
                            />
                            <span>{algo.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}</span>
                          </label>
                        ))}
                      </>
                    )}
                  </div>
                </div>

                {/* Hyperparameters & Sliders */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2 border-t border-slate-800">
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-2">
                      <span>Held-out Test Split</span>
                      <span className="font-mono text-indigo-400">{Math.round(testSplit * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.4"
                      step="0.05"
                      value={testSplit}
                      onChange={(e) => setTestSplit(parseFloat(e.target.value))}
                      className="w-full accent-indigo-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-300 mb-2">
                      <span>Cross-Validation Folds</span>
                      <span className="font-mono text-indigo-400">{cvFolds} Folds</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="6"
                      step="1"
                      value={cvFolds}
                      onChange={(e) => setCvFolds(parseInt(e.target.value))}
                      className="w-full accent-indigo-500"
                    />
                  </div>

                  <div>
                    <div className="text-xs font-medium text-slate-300 mb-2">Random Seed</div>
                    <div className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-300 font-mono">
                      42 (Deterministic)
                    </div>
                  </div>
                </div>

                <div className="flex justify-end pt-2">
                  <button
                    onClick={runTraining}
                    disabled={isTraining || selectedAlgorithms.length === 0}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50 transition cursor-pointer shadow-lg shadow-indigo-600/30"
                  >
                    {isTraining ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Training Pipeline Running...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        <span>Execute Training & Logging</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Training Logs Stream */}
              {trainingLogs.length > 0 && (
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 space-y-1">
                  <div className="text-indigo-400 font-bold mb-2 flex items-center gap-2">
                    <Terminal className="w-4 h-4" />
                    <span>ML Pipeline Execution Stream</span>
                  </div>
                  {trainingLogs.map((log, idx) => (
                    <div key={idx} className="leading-relaxed">
                      {log}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TAB 4: EXPERIMENTS & MLFLOW */}
          {activeTab === 'experiments' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">Experiment Tracking & Leaderboard</h1>
                <p className="text-slate-400 text-sm mt-1">
                  Review logged runs, compare holdout metrics, inspect parameters, and identify qualified candidates.
                </p>
              </div>

              {/* Experiments Table */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
                <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                  <h3 className="font-semibold text-sm text-slate-200">Run Comparison Matrix</h3>
                  <span className="text-xs text-slate-400 font-mono">Backend: MLflow SQLite Store</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900 text-slate-400 font-mono border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-3">Model Candidate</th>
                        <th className="px-4 py-3">Task</th>
                        <th className="px-4 py-3">Version</th>
                        <th className="px-4 py-3">Serving Status</th>
                        <th className="px-4 py-3">Holdout Metrics</th>
                        <th className="px-4 py-3">CV Mean</th>
                        <th className="px-4 py-3">Duration</th>
                        <th className="px-4 py-3">Created</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {models.map((m) => (
                        <tr key={m.id} className="hover:bg-slate-800/40">
                          <td className="px-4 py-3 font-semibold text-slate-200">
                            <div>{m.name}</div>
                            <span className="text-[10px] text-slate-400">{m.runId}</span>
                          </td>
                          <td className="px-4 py-3 text-slate-400">{m.taskType}</td>
                          <td className="px-4 py-3">v{m.version}</td>
                          <td className="px-4 py-3">
                            {m.isActive ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                                🟢 ACTIVE
                              </span>
                            ) : (
                              <span className="text-[11px] text-slate-400">Candidate</span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            {Object.entries(m.metrics).map(([k, v]) => (
                              <div key={k} className="text-[11px]">
                                <span className="text-slate-400">{k.toUpperCase()}: </span>
                                <span className="text-indigo-300 font-bold">{v}</span>
                              </div>
                            ))}
                          </td>
                          <td className="px-4 py-3 text-cyan-400 font-bold">{m.cvMean || '—'}</td>
                          <td className="px-4 py-3 text-slate-400">{m.durationSec}s</td>
                          <td className="px-4 py-3 text-slate-400">{m.createdAt.slice(0, 10)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: MODEL REGISTRY */}
          {activeTab === 'registry' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">Model Registry & Serving Promotion</h1>
                <p className="text-slate-400 text-sm mt-1">
                  Immutable versioned model artifacts. Promote qualified models to live serving with automated demotion.
                </p>
              </div>

              {/* Active Serving Alert Banner */}
              <div className="p-5 rounded-2xl bg-indigo-950/40 border border-indigo-800/50 flex flex-wrap items-center justify-between gap-4">
                <div>
                  <div className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                    Current Active Serving Model
                  </div>
                  <div className="text-lg font-bold text-white mt-1">
                    {activeModel.name} (Version {activeModel.version})
                  </div>
                  <div className="text-xs text-slate-300 mt-1 font-mono">
                    Model ID: {activeModel.id} | Task: {activeModel.taskType}
                  </div>
                </div>
                <div className="text-xs text-slate-400 text-right">
                  <div>Promoted by: <span className="text-slate-200">MLOps Operator</span></div>
                  <div>Timestamp: <span className="text-slate-200">{activeModel.activatedAt?.slice(0, 19) || 'Active'}</span></div>
                </div>
              </div>

              {/* Registered Model Cards */}
              <div className="space-y-4">
                {models.map((m) => (
                  <div
                    key={m.id}
                    className={`p-6 rounded-2xl border transition ${
                      m.isActive ? 'bg-slate-900 border-indigo-500/50' : 'bg-slate-900/50 border-slate-800'
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-3">
                          <h3 className="text-base font-bold text-white">{m.name}</h3>
                          <span className="px-2 py-0.5 rounded text-xs font-mono font-semibold bg-slate-800 text-slate-300">
                            v{m.version}
                          </span>
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                            m.status === 'validated' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'
                          }`}>
                            {m.status.toUpperCase()}
                          </span>
                        </div>
                        <div className="text-xs text-slate-400 font-mono mt-1">
                          Artifact: /artifacts/models/{m.algorithm}_{m.runId}.joblib
                        </div>
                      </div>

                      <div>
                        {m.isActive ? (
                          <span className="px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                            Active Serving Target
                          </span>
                        ) : (
                          <button
                            onClick={() => {
                              setModels((prev) =>
                                prev.map((item) => ({
                                  ...item,
                                  isActive: item.id === m.id,
                                  activatedAt: item.id === m.id ? new Date().toISOString() : item.activatedAt,
                                }))
                              );
                            }}
                            className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer transition shadow-md shadow-indigo-600/20"
                          >
                            Promote to Active Serving
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-800/80 text-xs font-mono">
                      {Object.entries(m.metrics).map(([k, v]) => (
                        <div key={k}>
                          <div className="text-slate-400">{k.toUpperCase()}</div>
                          <div className="text-slate-200 font-bold mt-0.5">{v}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 6: PREDICTION PLAYGROUND */}
          {activeTab === 'predictions' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">Real-Time Prediction Serving Playground</h1>
                <p className="text-slate-400 text-sm mt-1">
                  Send live HTTP inference payloads to active model <strong className="text-slate-200">{activeModel.name} (v{activeModel.version})</strong>.
                </p>
              </div>

              {/* Dynamic Feature Inputs Form */}
              <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-6">
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <h3 className="font-semibold text-sm text-slate-200">Model Feature Input Schema</h3>
                  <span className="text-xs font-mono text-indigo-400">Target: {activeModel.targetCol}</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
                  {activeModel.taskType === 'regression' ? (
                    <>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Square Feet (sqft)</label>
                        <input
                          type="number"
                          value={predictionInputs.sqft || 2400}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, sqft: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Bedrooms</label>
                        <input
                          type="number"
                          value={predictionInputs.beds || 3}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, beds: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Bathrooms</label>
                        <input
                          type="number"
                          step="0.5"
                          value={predictionInputs.baths || 2.5}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, baths: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Year Built</label>
                        <input
                          type="number"
                          value={predictionInputs.year_built || 2005}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, year_built: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Neighborhood</label>
                        <select
                          value={predictionInputs.neighborhood || 'Downtown'}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, neighborhood: e.target.value })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        >
                          <option value="Downtown">Downtown</option>
                          <option value="Suburbs">Suburbs</option>
                          <option value="Uptown">Uptown</option>
                          <option value="Lakeside">Lakeside</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Garage Spaces</label>
                        <input
                          type="number"
                          value={predictionInputs.garage || 2}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, garage: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Tenure Months</label>
                        <input
                          type="number"
                          value={predictionInputs.tenure_months || 12}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, tenure_months: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Monthly Charges ($)</label>
                        <input
                          type="number"
                          value={predictionInputs.monthly_charges || 75.0}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, monthly_charges: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Support Calls</label>
                        <input
                          type="number"
                          value={predictionInputs.support_calls || 2}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, support_calls: Number(e.target.value) })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Contract Type</label>
                        <select
                          value={predictionInputs.contract || 'Month-to-month'}
                          onChange={(e) => setPredictionInputs({ ...predictionInputs, contract: e.target.value })}
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono"
                        >
                          <option value="Month-to-month">Month-to-month</option>
                          <option value="One year">One year</option>
                          <option value="Two year">Two year</option>
                        </select>
                      </div>
                    </>
                  )}
                </div>

                <div className="flex justify-end pt-2">
                  <button
                    onClick={handlePredict}
                    disabled={isPredicting}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white cursor-pointer transition shadow-lg shadow-emerald-600/25"
                  >
                    <Zap className="w-4 h-4" />
                    <span>{isPredicting ? 'Inferring...' : 'Serve Model Prediction'}</span>
                  </button>
                </div>
              </div>

              {/* Output Result Card */}
              {latestPrediction && (
                <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 to-indigo-950/30 border border-indigo-500/40 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider font-mono">
                      Inference Response (200 OK)
                    </span>
                    <span className="text-xs text-slate-400 font-mono">
                      Latency: <strong className="text-emerald-400">{latestPrediction.latencyMs} ms</strong>
                    </span>
                  </div>

                  <div className="flex flex-wrap items-baseline gap-4">
                    <div className="text-3xl font-extrabold text-white font-mono">
                      {activeModel.taskType === 'regression' ? `$${Number(latestPrediction.prediction).toLocaleString()}` : (latestPrediction.prediction === 1 ? 'CHURN RISK (1)' : 'RETAINED (0)')}
                    </div>
                    <span className="text-xs text-slate-400 font-mono">Model: {latestPrediction.modelId} (v{latestPrediction.modelVersion})</span>
                  </div>

                  {latestPrediction.probabilities && (
                    <div className="space-y-1.5 pt-2">
                      <div className="text-xs text-slate-400">Class Probability Distribution:</div>
                      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                        {Object.entries(latestPrediction.probabilities).map(([cls, prob]) => (
                          <div key={cls} className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex justify-between">
                            <span className="text-slate-300">{cls}</span>
                            <span className="text-indigo-400 font-bold">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* TAB 7: MONITORING & DRIFT */}
          {activeTab === 'monitoring' && (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold text-white tracking-tight">Model Monitoring & Feature Drift</h1>
                  <p className="text-slate-400 text-sm mt-1">
                    Continuous statistical drift tests (Kolmogorov-Smirnov & Chi-Square), latency tracking, and operational incidents.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    onClick={handleExportDriftReport}
                    className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/20 cursor-pointer transition"
                    title="Export Current Drift Report as Downloadable JSON"
                  >
                    <Download className="w-4 h-4" />
                    <span>Export Drift Report (JSON)</span>
                  </button>

                  <button
                    onClick={handlePreviewDriftReport}
                    className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer transition shadow-sm"
                    title="Inspect and preview the full Drift Report JSON payload"
                  >
                    <FileCode className="w-4 h-4 text-cyan-400" />
                    <span>Preview JSON</span>
                  </button>

                  <button
                    onClick={handleSimulateDriftTraffic}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/30 cursor-pointer transition shadow-sm"
                  >
                    <AlertTriangle className="w-4 h-4 text-rose-400" />
                    <span>Simulate Shifted Traffic (Trigger Drift)</span>
                  </button>
                </div>
              </div>

              {/* Latency Quantiles Bar */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-[11px] text-slate-400">Requests</div>
                  <div className="text-lg font-bold text-white mt-1 font-mono">{telemetry.total}</div>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-[11px] text-slate-400">Avg Latency</div>
                  <div className="text-lg font-bold text-slate-200 mt-1 font-mono">{telemetry.avgLatency} ms</div>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-[11px] text-slate-400">p50 Latency</div>
                  <div className="text-lg font-bold text-emerald-400 mt-1 font-mono">{telemetry.p50} ms</div>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-[11px] text-slate-400">p95 Latency</div>
                  <div className="text-lg font-bold text-amber-400 mt-1 font-mono">{telemetry.p95} ms</div>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="text-[11px] text-slate-400">p99 Latency</div>
                  <div className="text-lg font-bold text-rose-400 mt-1 font-mono">{telemetry.p99} ms</div>
                </div>
              </div>

              {/* Historical Kolmogorov-Smirnov Test p-Values Line Chart */}
              <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-indigo-400" />
                      <h3 className="text-base font-bold text-white tracking-tight">
                        Historical Kolmogorov-Smirnov Test p-Values Over Time
                      </h3>
                      <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                        {activeModel.name} (v{activeModel.version})
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      Tracks empirical distribution divergence (p-values) across numerical features against baseline reference data. Any feature crossing below the dashed red line (α = 0.05) triggers a data drift alert.
                    </p>
                  </div>

                  {/* Actions & Controls */}
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      onClick={handleRunManualKSEval}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer transition shadow-sm"
                      title="Run a new KS evaluation batch against current traffic"
                    >
                      <RefreshCw className="w-3.5 h-3.5 text-indigo-400" />
                      <span>Run KS Test Batch</span>
                    </button>

                    <button
                      onClick={handleResetKSHistory}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer transition shadow-sm"
                      title="Reset historical evaluation series to initial baseline"
                    >
                      <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
                      <span>Reset History</span>
                    </button>

                    <button
                      onClick={handleExportDriftReport}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 cursor-pointer transition shadow-sm"
                      title="Download complete statistical drift report as JSON"
                    >
                      <Download className="w-3.5 h-3.5 text-indigo-400" />
                      <span>Export JSON</span>
                    </button>
                  </div>
                </div>

                {/* Feature Toggle Pill Bar */}
                <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-semibold text-slate-400 flex items-center gap-1">
                      <Filter className="w-3.5 h-3.5 text-slate-400" />
                      <span>Toggle Feature Lines:</span>
                    </span>
                    {activeNumericFeatures.map((feat, idx) => {
                      const isHidden = disabledChartFeatures[feat];
                      const color = FEATURE_COLORS[feat] || COLOR_PALETTE[idx % COLOR_PALETTE.length];
                      const latestVal = activeKSHistory[activeKSHistory.length - 1]?.[feat];
                      const isDrifted = typeof latestVal === 'number' && latestVal < 0.05;

                      return (
                        <button
                          key={feat}
                          onClick={() => toggleFeatureVisibility(feat)}
                          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono transition cursor-pointer border ${
                            isHidden
                              ? 'bg-slate-900/60 text-slate-500 border-slate-800 line-through opacity-60'
                              : 'bg-slate-900 text-slate-200 border-slate-700 hover:border-slate-600 shadow-sm'
                          }`}
                        >
                          <span
                            className="w-2.5 h-2.5 rounded-full shrink-0"
                            style={{ backgroundColor: isHidden ? '#475569' : color }}
                          />
                          <span>{feat}</span>
                          {!isHidden && typeof latestVal === 'number' && (
                            <span className={`text-[10px] font-bold px-1 rounded ${
                              isDrifted ? 'bg-rose-500/20 text-rose-400' : 'text-slate-400'
                            }`}>
                              p={latestVal.toFixed(3)}
                            </span>
                          )}
                          {isHidden ? <EyeOff className="w-3 h-3 text-slate-500 ml-0.5" /> : <Eye className="w-3 h-3 text-slate-400 ml-0.5" />}
                        </button>
                      );
                    })}
                  </div>

                  <div className="flex items-center gap-2 text-xs">
                    <button
                      onClick={() => setDisabledChartFeatures({})}
                      className="text-xs text-indigo-400 hover:text-indigo-300 font-semibold cursor-pointer"
                    >
                      Show All
                    </button>
                    <span className="text-slate-600">|</span>
                    <button
                      onClick={() => {
                        const allDisabled: Record<string, boolean> = {};
                        activeNumericFeatures.forEach((f) => (allDisabled[f] = true));
                        setDisabledChartFeatures(allDisabled);
                      }}
                      className="text-xs text-slate-400 hover:text-slate-300 cursor-pointer"
                    >
                      Hide All
                    </button>
                  </div>
                </div>

                {/* Recharts Multi-Line Graph */}
                <div className="h-[360px] w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={activeKSHistory}
                      margin={{ top: 20, right: 30, left: 10, bottom: 10 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis
                        dataKey="time"
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'monospace' }}
                        tickLine={{ stroke: '#334155' }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        ticks={[0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0]}
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'monospace' }}
                        tickLine={{ stroke: '#334155' }}
                        tickFormatter={(val) => (val === 0.05 ? '0.05 (α)' : val.toFixed(2))}
                      />
                      <Tooltip content={<CustomKSTooltip />} />
                      <Legend
                        wrapperStyle={{ paddingTop: '14px' }}
                        formatter={(value) => <span className="text-xs font-mono text-slate-300">{value}</span>}
                      />
                      {/* Significance Alert Threshold Line */}
                      <ReferenceLine
                        y={0.05}
                        stroke="#f43f5e"
                        strokeDasharray="5 5"
                        strokeWidth={2}
                        label={{
                          value: 'Drift Alert Threshold (α = 0.05)',
                          fill: '#fb7185',
                          position: 'insideTopRight',
                          fontSize: 11,
                          fontFamily: 'monospace',
                          fontWeight: 700,
                        }}
                      />
                      {/* Render visible feature lines */}
                      {visibleChartFeatures.map((featKey, idx) => {
                        const color = FEATURE_COLORS[featKey] || COLOR_PALETTE[idx % COLOR_PALETTE.length];
                        return (
                          <Line
                            key={featKey}
                            type="monotone"
                            dataKey={featKey}
                            name={featKey}
                            stroke={color}
                            strokeWidth={2.5}
                            dot={{ r: 4, strokeWidth: 1.5, fill: '#0f172a', stroke: color }}
                            activeDot={{ r: 6, stroke: '#ffffff', strokeWidth: 2 }}
                            isAnimationActive={true}
                          />
                        );
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Statistical Drift Insights Summary */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-3 border-t border-slate-800 text-xs">
                  <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/90">
                    <div className="text-slate-400">Current Lowest p-Value</div>
                    {(() => {
                      const latest = activeKSHistory[activeKSHistory.length - 1] || {};
                      let minVal = 1.0;
                      let minFeat = 'None';
                      activeNumericFeatures.forEach((f) => {
                        if (typeof latest[f] === 'number' && latest[f] < minVal) {
                          minVal = latest[f];
                          minFeat = f;
                        }
                      });
                      const isDrifted = minVal < 0.05;
                      return (
                        <div className="mt-1">
                          <div className={`text-base font-bold font-mono ${isDrifted ? 'text-rose-400' : 'text-emerald-400'}`}>
                            {minVal.toFixed(4)} ({minFeat})
                          </div>
                          <div className="text-[11px] text-slate-400 mt-0.5">
                            {isDrifted ? '⚠️ Below α=0.05 threshold' : '✅ In distribution'}
                          </div>
                        </div>
                      );
                    })()}
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/90">
                    <div className="text-slate-400">Drifted Features Status</div>
                    {(() => {
                      const latest = activeKSHistory[activeKSHistory.length - 1] || {};
                      const drifted = activeNumericFeatures.filter((f) => typeof latest[f] === 'number' && latest[f] < 0.05);
                      return (
                        <div className="mt-1">
                          <div className={`text-base font-bold font-mono ${drifted.length > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                            {drifted.length} of {activeNumericFeatures.length} Drifted
                          </div>
                          <div className="text-[11px] text-slate-400 mt-0.5 truncate">
                            {drifted.length > 0 ? drifted.join(', ') : 'All features stable'}
                          </div>
                        </div>
                      );
                    })()}
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/90">
                    <div className="text-slate-400">Hypothesis Test Method</div>
                    <div className="mt-1">
                      <div className="text-sm font-bold text-slate-200">Two-Sample KS Test</div>
                      <div className="text-[11px] text-slate-400 mt-0.5 font-mono">D = sup |F_ref(x) - F_prod(x)|</div>
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/90">
                    <div className="text-slate-400">Total Evaluation Series</div>
                    <div className="mt-1">
                      <div className="text-base font-bold text-white font-mono">{activeKSHistory.length} Batches</div>
                      <div className="text-[11px] text-indigo-400 mt-0.5 font-mono">Window: Last {activeKSHistory.length} points</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Feature Drift Evaluation Table & Export Summary */}
              <div className="rounded-2xl border border-slate-800 bg-slate-900/60 overflow-hidden">
                <div className="p-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 bg-slate-900/90">
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4 text-indigo-400" />
                    <div>
                      <h3 className="font-semibold text-sm text-slate-200">
                        Current Kolmogorov-Smirnov Feature Drift Matrix
                      </h3>
                      <p className="text-[11px] text-slate-400">
                        Statistical two-sample KS hypothesis testing: live production window vs reference training baseline.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleExportDriftReport}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer transition shadow-md shadow-indigo-600/20"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download JSON</span>
                    </button>
                    <button
                      onClick={handlePreviewDriftReport}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer transition"
                    >
                      <FileCode className="w-3.5 h-3.5 text-cyan-400" />
                      <span>View Raw Payload</span>
                    </button>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300 font-mono">
                    <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-3">Feature Name</th>
                        <th className="px-4 py-3">Type</th>
                        <th className="px-4 py-3">Test Method</th>
                        <th className="px-4 py-3">KS Statistic (D)</th>
                        <th className="px-4 py-3">p-Value</th>
                        <th className="px-4 py-3">Threshold (α)</th>
                        <th className="px-4 py-3">Baseline Mean / Std</th>
                        <th className="px-4 py-3">Production Mean / Std</th>
                        <th className="px-4 py-3">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {activeNumericFeatures.map((feat) => {
                        const latestKS = activeKSHistory[activeKSHistory.length - 1] || {};
                        const pVal = typeof latestKS[feat] === 'number' ? latestKS[feat] : 0.85;
                        const isDrifted = pVal < 0.05;
                        const refStats =
                          (activeModel.referenceDataSummary && activeModel.referenceDataSummary[feat]) ||
                          currentDataset.columns.find((c) => c.name === feat)?.stats || {
                            mean: 100,
                            std: 15,
                            median: 95,
                          };
                        const shiftFactor = isDrifted ? 1.45 : 1.02;
                        const currMean = Math.round(refStats.mean * shiftFactor * 100) / 100;
                        const currStd = Math.round(refStats.std * (isDrifted ? 1.3 : 1.0) * 100) / 100;
                        const dStat = isDrifted ? 0.642 : 0.085;

                        return (
                          <tr key={feat} className="hover:bg-slate-800/40 transition">
                            <td className="px-4 py-3 font-bold text-slate-200">
                              <div className="flex items-center gap-1.5">
                                <span
                                  className="w-2 h-2 rounded-full"
                                  style={{ backgroundColor: FEATURE_COLORS[feat] || '#6366f1' }}
                                />
                                <span>{feat}</span>
                              </div>
                            </td>
                            <td className="px-4 py-3 text-slate-400">continuous</td>
                            <td className="px-4 py-3 text-slate-400">Two-Sample KS</td>
                            <td className="px-4 py-3 text-slate-200 font-bold">{dStat.toFixed(3)}</td>
                            <td className="px-4 py-3">
                              <span
                                className={`font-bold ${
                                  isDrifted ? 'text-rose-400 bg-rose-500/20 px-1.5 py-0.5 rounded' : 'text-emerald-400'
                                }`}
                              >
                                {pVal.toFixed(4)}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-slate-400">0.05</td>
                            <td className="px-4 py-3 text-slate-300">
                              μ={refStats.mean} / σ={refStats.std}
                            </td>
                            <td className="px-4 py-3 text-slate-300">
                              μ={currMean} / σ={currStd}
                            </td>
                            <td className="px-4 py-3">
                              {isDrifted ? (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-sans font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">
                                  ⚠️ DRIFT DETECTED
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-sans font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                  STABLE
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Export Footer Bar */}
                <div className="p-4 bg-slate-950/80 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
                  <div className="text-slate-400">
                    Active Model: <span className="text-indigo-300 font-bold">{activeModel.name} (v{activeModel.version})</span> | Total Production Evaluated: <span className="text-white font-bold">{predictionLogs.length} inferences</span>
                  </div>
                  <div className="flex items-center gap-3 font-sans">
                    <button
                      onClick={handleExportDriftReport}
                      className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer transition shadow-md shadow-indigo-600/20"
                    >
                      <Download className="w-4 h-4" />
                      <span>Export Full Drift Report as JSON</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Active Drift Incident Alerts */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-rose-400" />
                  <span>Operational Incident Center</span>
                </h3>
                {alerts.length === 0 ? (
                  <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 text-center text-xs text-slate-400">
                    Zero alerts recorded. Model operating within latency and drift tolerances.
                  </div>
                ) : (
                  alerts.map((a) => (
                    <div
                      key={a.id}
                      className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-4 text-xs ${
                        a.status === 'ACTIVE'
                          ? 'bg-rose-950/20 border-rose-900/50 text-rose-200'
                          : 'bg-slate-900/40 border-slate-800 text-slate-400'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-rose-400">[{a.type}]</span>
                          <span className="font-semibold">{a.message}</span>
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                            {a.status}
                          </span>
                        </div>
                        <p className="text-slate-400 text-[11px]">{a.details}</p>
                      </div>

                      {a.status === 'ACTIVE' && (
                        <button
                          onClick={() => {
                            setAlerts((prev) =>
                              prev.map((item) => (item.id === a.id ? { ...item, status: 'RESOLVED' } : item))
                            );
                          }}
                          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium cursor-pointer transition text-xs"
                        >
                          Resolve Alert
                        </button>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* TAB 8: API DOCS & ARCHITECTURE */}
          {activeTab === 'apiDocs' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">API Documentation & System Design</h1>
                <p className="text-slate-400 text-sm mt-1">
                  Standard OpenAPI contracts, production deployment commands, and architectural interview specs.
                </p>
              </div>

              {/* Endpoints List */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-slate-200">FastAPI Production Endpoints</h3>
                <div className="space-y-2 text-xs font-mono">
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">GET</span>
                      <span className="text-slate-200">/health &amp; /ready</span>
                    </div>
                    <span className="text-slate-400">Liveness and readiness container health probes</span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-bold">POST</span>
                      <span className="text-slate-200">/api/v1/datasets/upload</span>
                    </div>
                    <span className="text-slate-400">CSV schema validation &amp; leakage diagnostics</span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-bold">POST</span>
                      <span className="text-slate-200">/api/v1/training/jobs</span>
                    </div>
                    <span className="text-slate-400">Asynchronous / bounded ML training with MLflow</span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-bold">POST</span>
                      <span className="text-slate-200">/api/v1/predict</span>
                    </div>
                    <span className="text-slate-400">Schema-validated real-time model inference</span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">GET</span>
                      <span className="text-slate-200">/api/v1/monitoring/drift</span>
                    </div>
                    <span className="text-slate-400">Kolmogorov-Smirnov two-sample drift calculations</span>
                  </div>
                </div>
              </div>

              {/* Startup Commands */}
              <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-3 font-mono text-xs">
                <div className="text-slate-300 font-bold text-sm">Local Terminal Commands</div>
                <div className="p-3 rounded-lg bg-slate-950 text-slate-300 space-y-1">
                  <div className="text-slate-500"># Start FastAPI Backend</div>
                  <div>uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload</div>
                  <div className="text-slate-500 pt-2"># Start Streamlit UI Dashboard</div>
                  <div>streamlit run app/dashboard/Home.py --server.port 8501</div>
                  <div className="text-slate-500 pt-2"># Start Full Multi-Service Stack with Docker Compose</div>
                  <div>docker-compose up --build -d</div>
                  <div className="text-slate-500 pt-2"># Run Automated Test Suite</div>
                  <div>pytest tests/ -v</div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* JSON Drift Report Inspection Modal */}
      {jsonModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="w-full max-w-4xl max-h-[88vh] rounded-2xl bg-slate-900 border border-slate-700 shadow-2xl flex flex-col overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/80">
              <div className="flex items-center gap-2.5">
                <FileCode className="w-5 h-5 text-indigo-400" />
                <div>
                  <h3 className="font-bold text-sm text-slate-100">
                    Production Drift Report JSON Payload
                  </h3>
                  <p className="text-[11px] text-slate-400 font-mono">
                    Model: {activeModel.name} (v{activeModel.version}) | Schema: Kolmogorov-Smirnov v1.0
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyJson}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer transition"
                  title="Copy JSON to clipboard"
                >
                  {copiedJson ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400 font-bold">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5 text-slate-300" />
                      <span>Copy JSON</span>
                    </>
                  )}
                </button>
                <button
                  onClick={handleExportDriftReport}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer transition shadow-md shadow-indigo-600/20"
                  title="Download as JSON file"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download .json</span>
                </button>
                <button
                  onClick={() => setJsonModalOpen(false)}
                  className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition cursor-pointer ml-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-auto p-4 bg-slate-950 font-mono text-xs text-indigo-200/90 leading-relaxed">
              <pre className="whitespace-pre">{jsonExportContent}</pre>
            </div>

            <div className="p-3 border-t border-slate-800 bg-slate-900/90 flex items-center justify-between text-xs font-mono text-slate-400">
              <div>
                Report Timestamp: {new Date().toISOString().slice(0, 19)}Z
              </div>
              <button
                onClick={() => setJsonModalOpen(false)}
                className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-sans font-medium cursor-pointer transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Notification Toast */}
      {exportToastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl bg-slate-900 border border-emerald-500/40 text-emerald-300 text-xs font-mono shadow-2xl backdrop-blur-md">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{exportToastMessage}</span>
        </div>
      )}
    </div>
  );
}
