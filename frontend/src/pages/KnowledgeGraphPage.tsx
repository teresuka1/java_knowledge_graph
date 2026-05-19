import {useEffect, useMemo, useState} from 'react';
import type {ComponentType} from 'react';
import {
  AlertCircle,
  Database,
  GitBranch,
  Info,
  Loader2,
  Network,
  RefreshCw,
  Search,
  UploadCloud,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import {knowledgeGraphApi} from '../api/knowledgeGraph';
import {getErrorMessage} from '../api/request';
import type {GraphEdge, GraphNode, GraphNodeDetail, GraphResponse, GraphStats} from '../types/knowledgeGraph';

interface PositionedNode extends GraphNode {
  x: number;
  y: number;
  radius: number;
}

interface DomainCluster {
  domain: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

const DOMAIN_COLORS: Record<string, string> = {
  java: '#2563eb',
  mysql: '#16a34a',
  JVM: '#dc2626',
  os: '#7c3aed',
  cn: '#0891b2',
  data_structure: '#ea580c',
};

const RELATION_LABELS: Record<string, string> = {
  depends_on: '依赖',
  has_part: '包含',
  used_for: '用于',
  related_to: '相关',
  compare_with: '对比',
};

export default function KnowledgeGraphPage() {
  const [graph, setGraph] = useState<GraphResponse>({nodes: [], edges: []});
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNodeDetail | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const [domain, setDomain] = useState('');
  const [keyword, setKeyword] = useState('');
  const [depth, setDepth] = useState(1);
  const [limit, setLimit] = useState(120);
  const [spacingScale, setSpacingScale] = useState(1);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');

  const loadStats = async () => {
    try {
      setStats(await knowledgeGraphApi.getStats());
    } catch {
      setStats(null);
    }
  };

  const loadGraph = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await knowledgeGraphApi.getGraph({
        domain: domain || undefined,
        keyword: keyword.trim() || undefined,
        depth,
        limit,
      });
      setGraph(data);
      setSelectedNode(null);
      setSelectedEdge(null);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
    loadGraph();
  }, []);

  const positionedNodes = useMemo(
    () => layoutNodes(graph.nodes, spacingScale),
    [graph.nodes, spacingScale]
  );
  const domainClusters = useMemo(() => buildDomainClusters(positionedNodes), [positionedNodes]);
  const nodeMap = useMemo(
    () => new Map(positionedNodes.map((node) => [node.id, node])),
    [positionedNodes]
  );

  const handleImport = async () => {
    setImporting(true);
    setError('');
    try {
      const result = await knowledgeGraphApi.importKnowledgeGraph();
      await loadStats();
      await loadGraph();
      setError(`导入完成：${result.nodeCount} 个节点，${result.edgeCount} 条关系`);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setImporting(false);
    }
  };

  const handleNodeClick = async (node: GraphNode) => {
    setSelectedEdge(null);
    try {
      setSelectedNode(await knowledgeGraphApi.getNodeDetail(node.domain, node.name));
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const domains = stats?.domains ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white">知识图谱</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            可视化查看后端面试知识点之间的依赖、组成和用途关系
          </p>
        </div>
        <button
          onClick={handleImport}
          disabled={importing}
          className="btn-primary inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm disabled:opacity-60"
        >
          {importing ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
          导入图谱
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <StatCard icon={Database} label="节点" value={stats?.nodeCount ?? 0} />
        <StatCard icon={GitBranch} label="关系" value={stats?.edgeCount ?? 0} />
        <StatCard icon={Network} label="领域" value={stats?.domainCount ?? 0} />
        <StatCard icon={Info} label="关系类型" value={stats?.relationTypes.length ?? 0} />
      </div>

      <div className="dark-card rounded-lg p-4">
        <div className="grid gap-3 xl:grid-cols-[1.4fr_0.8fr_0.6fr_0.6fr_auto]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              value={keyword}
              onChange={(event) => setKeyword(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  loadGraph();
                }
              }}
              className="dark-input h-11 w-full rounded-lg pl-10 pr-3 text-sm"
              placeholder="搜索知识点，例如 JVM、索引、HTTPS"
            />
          </label>
          <select
            value={domain}
            onChange={(event) => setDomain(event.target.value)}
            className="dark-input h-11 rounded-lg px-3 text-sm"
          >
            <option value="">全部领域</option>
            {domains.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
          <select
            value={depth}
            onChange={(event) => setDepth(Number(event.target.value))}
            className="dark-input h-11 rounded-lg px-3 text-sm"
          >
            <option value={0}>只看节点</option>
            <option value={1}>一度关系</option>
            <option value={2}>二度关系</option>
          </select>
          <input
            value={limit}
            min={50}
            max={500}
            step={10}
            type="number"
            onChange={(event) => setLimit(Number(event.target.value))}
            className="dark-input h-11 rounded-lg px-3 text-sm"
          />
          <button
            onClick={loadGraph}
            disabled={loading}
            className="btn-secondary inline-flex h-11 items-center justify-center gap-2 rounded-lg px-4 text-sm disabled:opacity-60"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            刷新
          </button>
        </div>
      </div>

      {error && (
        <div
          className={`flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50
            px-4 py-3 text-sm text-amber-800 dark:border-amber-900/60
            dark:bg-amber-950/40 dark:text-amber-200`}
        >
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
        <div className="dark-card min-h-[620px] rounded-lg p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white">图谱画布</h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                当前展示 {graph.nodes.length} 个节点，{graph.edges.length} 条关系
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setSpacingScale((value) => Math.max(0.75, value - 0.15))}
                className="btn-secondary inline-flex h-9 w-9 items-center justify-center rounded-lg"
                title="缩小间距"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              <button
                onClick={() => setSpacingScale((value) => Math.min(1.8, value + 0.15))}
                className="btn-secondary inline-flex h-9 w-9 items-center justify-center rounded-lg"
                title="放大间距"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
            </div>
          </div>

          <GraphCanvas
            nodes={positionedNodes}
            edges={graph.edges}
            domainClusters={domainClusters}
            nodeMap={nodeMap}
            selectedNodeId={selectedNode?.node.id}
            selectedEdgeId={selectedEdge?.id}
            hoveredNodeId={hoveredNodeId}
            hoveredEdgeId={hoveredEdgeId}
            onNodeClick={handleNodeClick}
            onNodeHover={setHoveredNodeId}
            onEdgeClick={(edge) => {
              setSelectedEdge(edge);
              setSelectedNode(null);
            }}
            onEdgeHover={setHoveredEdgeId}
          />
        </div>

        <DetailPanel selectedNode={selectedNode} selectedEdge={selectedEdge} />
      </div>
    </div>
  );
}

function GraphCanvas({
  nodes,
  edges,
  domainClusters,
  nodeMap,
  selectedNodeId,
  selectedEdgeId,
  hoveredNodeId,
  hoveredEdgeId,
  onNodeClick,
  onNodeHover,
  onEdgeClick,
  onEdgeHover,
}: {
  nodes: PositionedNode[];
  edges: GraphEdge[];
  domainClusters: DomainCluster[];
  nodeMap: Map<string, PositionedNode>;
  selectedNodeId?: string;
  selectedEdgeId?: string;
  hoveredNodeId: string | null;
  hoveredEdgeId: string | null;
  onNodeClick: (node: GraphNode) => void;
  onNodeHover: (nodeId: string | null) => void;
  onEdgeClick: (edge: GraphEdge) => void;
  onEdgeHover: (edgeId: string | null) => void;
}) {
  if (nodes.length === 0) {
    return (
      <div
        className={`flex h-[540px] items-center justify-center rounded-lg border border-dashed
          border-slate-300 text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400`}
      >
        暂无图谱数据，请先点击“导入图谱”或调整搜索条件
      </div>
    );
  }

  return (
    <svg
      viewBox="0 0 900 900"
      className="h-[540px] w-full rounded-lg border border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-slate-950"
      role="img"
      aria-label="知识图谱可视化画布"
    >
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
        </marker>
      </defs>
      {domainClusters.map((cluster) => (
        <g key={cluster.domain}>
          <rect
            x={cluster.x}
            y={cluster.y}
            width={cluster.width}
            height={cluster.height}
            rx={24}
            fill={DOMAIN_COLORS[cluster.domain] ?? '#64748b'}
            opacity={0.07}
          />
          <text
            x={cluster.x + 18}
            y={cluster.y + 28}
            className="fill-slate-500 text-[15px] font-semibold dark:fill-slate-300"
          >
            {formatDomainName(cluster.domain)}
          </text>
        </g>
      ))}
      {edges.map((edge) => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target) {
          return null;
        }
        const selected = edge.id === selectedEdgeId;
        const hovered = edge.id === hoveredEdgeId;
        const relatedToFocusedNode = edge.source === hoveredNodeId
          || edge.target === hoveredNodeId
          || edge.source === selectedNodeId
          || edge.target === selectedNodeId;
        const emphasize = selected || hovered || relatedToFocusedNode;
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;
        return (
          <g
            key={edge.id}
            className="cursor-pointer"
            onClick={() => onEdgeClick(edge)}
            onMouseEnter={() => onEdgeHover(edge.id)}
            onMouseLeave={() => onEdgeHover(null)}
          >
            <line
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={selected ? '#f97316' : emphasize ? '#64748b' : '#cbd5e1'}
              strokeWidth={selected ? 3 : emphasize ? 2 : 1.2}
              opacity={emphasize ? 0.95 : 0.55}
              markerEnd="url(#arrow)"
            />
            {(selected || hovered || nodes.length <= 35) && (
              <text
                x={midX}
                y={midY - 6}
                textAnchor="middle"
                className="fill-slate-500 text-[11px] dark:fill-slate-300"
              >
                {RELATION_LABELS[edge.relation] ?? edge.relation}
              </text>
            )}
          </g>
        );
      })}
      {nodes.map((node) => {
        const selected = node.id === selectedNodeId;
        const hovered = node.id === hoveredNodeId;
        const color = DOMAIN_COLORS[node.domain] ?? '#64748b';
        const showLabel = shouldShowNodeLabel(node, nodes.length, selected, hovered);
        return (
          <g
            key={node.id}
            className="cursor-pointer"
            onClick={() => onNodeClick(node)}
            onMouseEnter={() => onNodeHover(node.id)}
            onMouseLeave={() => onNodeHover(null)}
          >
            <circle
              cx={node.x}
              cy={node.y}
              r={node.radius}
              fill={color}
              opacity={selected ? 1 : 0.86}
              stroke={selected ? '#facc15' : '#ffffff'}
              strokeWidth={selected ? 5 : 2}
            />
            <title>{node.name}</title>
            {showLabel && (
              <g>
                <rect
                  x={node.x - labelWidth(shortLabel(node.name)) / 2}
                  y={node.y + node.radius + 8}
                  width={labelWidth(shortLabel(node.name))}
                  height={22}
                  rx={6}
                  fill="white"
                  opacity={selected || hovered ? 0.96 : 0.88}
                />
                <text
                  x={node.x}
                  y={node.y + node.radius + 23}
                  textAnchor="middle"
                  className="fill-slate-700 text-[12px] font-semibold dark:fill-slate-100"
                >
                  {shortLabel(node.name)}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function DetailPanel({
  selectedNode,
  selectedEdge,
}: {
  selectedNode: GraphNodeDetail | null;
  selectedEdge: GraphEdge | null;
}) {
  return (
    <aside className="dark-card rounded-lg p-5">
      <h2 className="mb-4 text-lg font-semibold text-slate-900 dark:text-white">详情</h2>
      {!selectedNode && !selectedEdge && (
        <p className="text-sm leading-6 text-slate-500 dark:text-slate-400">
          点击画布中的节点或关系，可以查看来源、证据句子和关联知识点。
        </p>
      )}

      {selectedNode && (
        <div className="space-y-4">
          <div>
            <p className="text-xs text-slate-400">知识点</p>
            <h3 className="mt-1 text-xl font-semibold text-slate-900 dark:text-white">{selectedNode.node.name}</h3>
          </div>
          <InfoRow label="类型" value={selectedNode.node.type ?? '-'} />
          <InfoRow label="领域" value={selectedNode.node.domain} />
          <InfoRow label="来源" value={selectedNode.node.sourceFile ?? '-'} />
          <InfoRow label="出现次数" value={selectedNode.node.mentionCount.toString()} />
          <div>
            <p className="mb-2 text-xs text-slate-400">相关关系</p>
            <div className="max-h-[360px] space-y-2 overflow-y-auto pr-1 scrollbar-thin">
              {selectedNode.relations.map((edge) => (
                <div key={edge.id} className="rounded-lg border border-slate-200 p-3 text-sm dark:border-slate-700">
                  <p className="font-medium text-slate-800 dark:text-slate-100">
                    {edge.source} → {RELATION_LABELS[edge.relation] ?? edge.relation} → {edge.target}
                  </p>
                  {edge.evidence && (
                    <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">{edge.evidence}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {selectedEdge && (
        <div className="space-y-4">
          <InfoRow label="关系" value={RELATION_LABELS[selectedEdge.relation] ?? selectedEdge.relation} />
          <InfoRow label="起点" value={selectedEdge.source} />
          <InfoRow label="终点" value={selectedEdge.target} />
          <InfoRow label="置信度" value={selectedEdge.confidence?.toFixed(3) ?? '-'} />
          <InfoRow label="章节" value={selectedEdge.sectionTitle ?? '-'} />
          <InfoRow label="来源" value={selectedEdge.sourceFile ?? '-'} />
          <div>
            <p className="text-xs text-slate-400">证据句子</p>
            <p className="mt-2 rounded-lg bg-slate-100 p-3 text-sm leading-6 text-slate-700 dark:bg-slate-900 dark:text-slate-200">
              {selectedEdge.evidence ?? '暂无证据'}
            </p>
          </div>
        </div>
      )}
    </aside>
  );
}

function StatCard({icon: Icon, label, value}: {
  icon: ComponentType<{className?: string}>;
  label: string;
  value: number;
}) {
  return (
    <div className="dark-card rounded-lg p-4">
      <div className="flex items-center gap-3">
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-lg bg-primary-50
            text-primary-600 dark:bg-primary-900/40 dark:text-primary-300`}
        >
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-xs text-slate-500 dark:text-slate-400">{label}</p>
          <p className="text-xl font-bold text-slate-900 dark:text-white">{value}</p>
        </div>
      </div>
    </div>
  );
}

function InfoRow({label, value}: {label: string; value: string}) {
  return (
    <div>
      <p className="text-xs text-slate-400">{label}</p>
      <p className="mt-1 break-words text-sm font-medium text-slate-800 dark:text-slate-100">{value}</p>
    </div>
  );
}

function layoutNodes(nodes: GraphNode[], spacingScale: number): PositionedNode[] {
  const width = 900;
  const height = 900;
  const padding = 48;
  const grouped = new Map<string, GraphNode[]>();
  nodes.forEach((node) => {
    const group = grouped.get(node.domain) ?? [];
    group.push(node);
    grouped.set(node.domain, group);
  });

  const positioned: PositionedNode[] = [];
  const domains = Array.from(grouped.keys());
  const domainCenters = buildDomainCenters(domains, width, height, spacingScale);

  domains.forEach((domainName) => {
    const group = grouped.get(domainName) ?? [];
    const center = domainCenters.get(domainName) ?? {x: width / 2, y: height / 2};
    group.forEach((node, index) => {
      const angle = index * 2.399963229728653;
      const orbit = (34 + Math.sqrt(index) * 38) * spacingScale;
      positioned.push({
        ...node,
        x: center.x + Math.cos(angle) * orbit,
        y: center.y + Math.sin(angle) * orbit,
        radius: getNodeRadius(node),
      });
    });
  });

  relaxLayout(positioned, domainCenters, width, height, padding, spacingScale);
  return positioned;
}

function shortLabel(value: string): string {
  return value.length > 12 ? `${value.slice(0, 12)}...` : value;
}

function buildDomainCenters(
  domains: string[],
  width: number,
  height: number,
  spacingScale: number
) {
  const centers = new Map<string, {x: number; y: number}>();
  const columns = domains.length <= 3 ? domains.length : 3;
  const rows = Math.ceil(domains.length / Math.max(columns, 1));
  const centerX = width / 2;
  const centerY = height / 2;
  domains.forEach((domain, index) => {
    const column = index % Math.max(columns, 1);
    const row = Math.floor(index / Math.max(columns, 1));
    const baseX = ((column + 1) * width) / (columns + 1);
    const baseY = ((row + 1) * height) / (rows + 1);
    centers.set(domain, {
      x: clamp(centerX + (baseX - centerX) * spacingScale, 150, width - 150),
      y: clamp(centerY + (baseY - centerY) * spacingScale, 150, height - 150),
    });
  });
  return centers;
}

function getNodeRadius(node: GraphNode): number {
  return Math.min(30, Math.max(15, 14 + Math.log((node.mentionCount ?? 0) + 1) * 3.2));
}

function relaxLayout(
  nodes: PositionedNode[],
  centers: Map<string, {x: number; y: number}>,
  width: number,
  height: number,
  padding: number,
  spacingScale: number
) {
  for (let iteration = 0; iteration < 160; iteration++) {
    for (let i = 0; i < nodes.length; i++) {
      const first = nodes[i];
      const center = centers.get(first.domain);
      if (center) {
        first.x += (center.x - first.x) * 0.012;
        first.y += (center.y - first.y) * 0.012;
      }

      for (let j = i + 1; j < nodes.length; j++) {
        const second = nodes[j];
        const dx = second.x - first.x;
        const dy = second.y - first.y;
        const distance = Math.max(Math.hypot(dx, dy), 0.01);
        const minimumDistance = first.radius + second.radius + 24 * spacingScale;
        if (distance >= minimumDistance) {
          continue;
        }
        const overlap = (minimumDistance - distance) / 2;
        const nx = dx / distance;
        const ny = dy / distance;
        first.x -= nx * overlap;
        first.y -= ny * overlap;
        second.x += nx * overlap;
        second.y += ny * overlap;
      }
    }

    nodes.forEach((node) => {
      node.x = clamp(node.x, padding, width - padding);
      node.y = clamp(node.y, padding, height - padding);
    });
  }
}

function shouldShowNodeLabel(
  node: PositionedNode,
  nodeCount: number,
  selected: boolean,
  hovered: boolean
): boolean {
  if (selected || hovered) {
    return true;
  }
  if (nodeCount <= 40) {
    return true;
  }
  if (nodeCount <= 90) {
    return (node.mentionCount ?? 0) >= 6;
  }
  return (node.mentionCount ?? 0) >= 12;
}

function labelWidth(label: string): number {
  return Math.max(36, label.length * 12 + 16);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function buildDomainClusters(nodes: PositionedNode[]): DomainCluster[] {
  const grouped = new Map<string, PositionedNode[]>();
  nodes.forEach((node) => {
    const group = grouped.get(node.domain) ?? [];
    group.push(node);
    grouped.set(node.domain, group);
  });

  return Array.from(grouped.entries()).map(([domain, group]) => {
    const minX = Math.min(...group.map((node) => node.x - node.radius));
    const maxX = Math.max(...group.map((node) => node.x + node.radius));
    const minY = Math.min(...group.map((node) => node.y - node.radius));
    const maxY = Math.max(...group.map((node) => node.y + node.radius));
    return {
      domain,
      x: Math.max(18, minX - 36),
      y: Math.max(18, minY - 44),
      width: Math.min(864, maxX - minX + 72),
      height: Math.min(864, maxY - minY + 88),
    };
  });
}

function formatDomainName(domain: string): string {
  const labels: Record<string, string> = {
    cn: '计算机网络',
    data_structure: '数据结构',
    java: 'Java',
    JVM: 'JVM',
    mysql: 'MySQL',
    os: '操作系统',
  };
  return labels[domain] ?? domain;
}
