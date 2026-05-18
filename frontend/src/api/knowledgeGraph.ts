import {request} from './request';
import type {
  GraphImportResponse,
  GraphNode,
  GraphNodeDetail,
  GraphResponse,
  GraphStats,
} from '../types/knowledgeGraph';

export interface GraphQueryParams {
  domain?: string;
  keyword?: string;
  depth?: number;
  limit?: number;
}

function buildQuery(params: GraphQueryParams): string {
  const searchParams = new URLSearchParams();
  if (params.domain) {
    searchParams.append('domain', params.domain);
  }
  if (params.keyword) {
    searchParams.append('keyword', params.keyword);
  }
  if (params.depth !== undefined) {
    searchParams.append('depth', params.depth.toString());
  }
  if (params.limit !== undefined) {
    searchParams.append('limit', params.limit.toString());
  }
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
}

export const knowledgeGraphApi = {
  importKnowledgeGraph(): Promise<GraphImportResponse> {
    return request.post<GraphImportResponse>('/api/knowledgegraph/import');
  },

  getGraph(params: GraphQueryParams = {}): Promise<GraphResponse> {
    return request.get<GraphResponse>(`/api/knowledgegraph/graph${buildQuery(params)}`);
  },

  searchNodes(domain: string | undefined, keyword: string, limit = 20): Promise<GraphNode[]> {
    const params = new URLSearchParams({keyword, limit: limit.toString()});
    if (domain) {
      params.append('domain', domain);
    }
    return request.get<GraphNode[]>(`/api/knowledgegraph/nodes/search?${params.toString()}`);
  },

  getNodeDetail(domain: string, nodeName: string): Promise<GraphNodeDetail> {
    const params = new URLSearchParams({domain, nodeName});
    return request.get<GraphNodeDetail>(`/api/knowledgegraph/nodes/detail?${params.toString()}`);
  },

  getStats(): Promise<GraphStats> {
    return request.get<GraphStats>('/api/knowledgegraph/stats');
  },
};
