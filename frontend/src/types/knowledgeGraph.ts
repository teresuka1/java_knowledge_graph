export interface GraphNode {
  id: string;
  name: string;
  type: string | null;
  domain: string;
  mentionCount: number;
  sourceFile: string | null;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  confidence: number | null;
  evidence: string | null;
  sourceFile: string | null;
  sectionTitle: string | null;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphImportResponse {
  nodeCount: number;
  edgeCount: number;
  skippedCount: number;
}

export interface GraphNodeDetail {
  node: GraphNode;
  relations: GraphEdge[];
}

export interface GraphStats {
  nodeCount: number;
  edgeCount: number;
  domainCount: number;
  domains: string[];
  relationTypes: string[];
}
