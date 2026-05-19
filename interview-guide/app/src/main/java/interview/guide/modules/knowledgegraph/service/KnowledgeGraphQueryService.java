package interview.guide.modules.knowledgegraph.service;

import interview.guide.common.exception.BusinessException;
import interview.guide.common.exception.ErrorCode;
import interview.guide.modules.knowledgegraph.model.GraphEdgeDTO;
import interview.guide.modules.knowledgegraph.model.GraphNodeDTO;
import interview.guide.modules.knowledgegraph.model.GraphNodeDetailDTO;
import interview.guide.modules.knowledgegraph.model.GraphResponse;
import interview.guide.modules.knowledgegraph.model.GraphStatsDTO;
import interview.guide.modules.knowledgegraph.model.KnowledgeGraphEdgeEntity;
import interview.guide.modules.knowledgegraph.model.KnowledgeGraphNodeEntity;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphEdgeRepository;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphNodeRepository;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class KnowledgeGraphQueryService {

  private static final int DEFAULT_LIMIT = 200;
  private static final int MAX_LIMIT = 500;

  private final KnowledgeGraphNodeRepository nodeRepository;
  private final KnowledgeGraphEdgeRepository edgeRepository;

  @Transactional(readOnly = true)
  public GraphResponse getGraph(String domain, String keyword, Integer depth, Integer limit) {
    String normalizedDomain = normalize(domain);
    int safeLimit = clampLimit(limit);
    int safeDepth = depth == null ? 1 : Math.max(0, Math.min(depth, 2));

    List<KnowledgeGraphNodeEntity> seedNodes = isBlank(keyword)
        ? nodeRepository.findTopNodes(normalizedDomain, PageRequest.of(0, Math.min(safeLimit, 80)))
        : nodeRepository.searchNodes(normalizedDomain, keyword, PageRequest.of(0, 20));

    if (seedNodes.isEmpty()) {
      return new GraphResponse(List.of(), List.of());
    }

    Set<String> nodeNames = new LinkedHashSet<>();
    seedNodes.forEach(node -> nodeNames.add(node.getName()));
    List<KnowledgeGraphEdgeEntity> edges = new ArrayList<>();

    if (safeDepth > 0) {
      List<KnowledgeGraphEdgeEntity> firstLevelEdges = edgeRepository.findEdgesTouching(
          normalizedDomain,
          nodeNames,
          PageRequest.of(0, safeLimit)
      );
      edges.addAll(firstLevelEdges);
      firstLevelEdges.forEach(edge -> {
        nodeNames.add(edge.getSourceName());
        nodeNames.add(edge.getTargetName());
      });
    }

    if (safeDepth > 1 && nodeNames.size() < safeLimit) {
      List<KnowledgeGraphEdgeEntity> secondLevelEdges = edgeRepository.findEdgesTouching(
          normalizedDomain,
          nodeNames,
          PageRequest.of(0, safeLimit)
      );
      edges.addAll(secondLevelEdges);
      secondLevelEdges.forEach(edge -> {
        nodeNames.add(edge.getSourceName());
        nodeNames.add(edge.getTargetName());
      });
    }

    List<KnowledgeGraphNodeEntity> nodes = collectNodes(normalizedDomain, nodeNames, safeLimit);
    Set<String> allowedNames = new LinkedHashSet<>();
    nodes.forEach(node -> allowedNames.add(node.getName()));

    List<GraphEdgeDTO> graphEdges = dedupeEdges(edges).stream()
        .filter(edge -> allowedNames.contains(edge.getSourceName()))
        .filter(edge -> allowedNames.contains(edge.getTargetName()))
        .limit(safeLimit)
        .map(this::toEdgeDTO)
        .toList();

    return new GraphResponse(nodes.stream().map(this::toNodeDTO).toList(), graphEdges);
  }

  @Transactional(readOnly = true)
  public List<GraphNodeDTO> searchNodes(String domain, String keyword, Integer limit) {
    if (isBlank(keyword)) {
      return List.of();
    }
    return nodeRepository.searchNodes(
            normalize(domain),
            keyword,
            PageRequest.of(0, clampLimit(limit))
        )
        .stream()
        .map(this::toNodeDTO)
        .toList();
  }

  @Transactional(readOnly = true)
  public GraphNodeDetailDTO getNodeDetail(String domain, String nodeName) {
    String normalizedDomain = normalize(domain);
    KnowledgeGraphNodeEntity node = nodeRepository.findByDomainAndName(normalizedDomain, nodeName)
        .orElseThrow(() -> new BusinessException(
            ErrorCode.KNOWLEDGE_GRAPH_NODE_NOT_FOUND,
            "知识图谱节点不存在: " + nodeName
        ));

    List<GraphEdgeDTO> relations = edgeRepository.findEdgesTouching(
            normalizedDomain,
            List.of(nodeName),
            PageRequest.of(0, 100)
        )
        .stream()
        .map(this::toEdgeDTO)
        .toList();

    return new GraphNodeDetailDTO(toNodeDTO(node), relations);
  }

  @Transactional(readOnly = true)
  public GraphStatsDTO getStats() {
    List<String> domains = nodeRepository.findAllDomains();
    return new GraphStatsDTO(
        nodeRepository.count(),
        edgeRepository.count(),
        domains.size(),
        domains,
        edgeRepository.findAllRelationTypes()
    );
  }

  private List<KnowledgeGraphNodeEntity> collectNodes(
      String domain,
      Set<String> nodeNames,
      int limit
  ) {
    if (nodeNames.isEmpty()) {
      return List.of();
    }
    Map<String, KnowledgeGraphNodeEntity> nodes = new LinkedHashMap<>();
    List<KnowledgeGraphNodeEntity> matchedNodes = domain == null
        ? nodeRepository.findByNameIn(nodeNames)
        : nodeRepository.findByDomainAndNameIn(domain, nodeNames);
    matchedNodes.stream()
        .limit(limit)
        .forEach(node -> nodes.put(node.getDomain() + ":" + node.getName(), node));
    return List.copyOf(nodes.values());
  }

  private List<KnowledgeGraphEdgeEntity> dedupeEdges(List<KnowledgeGraphEdgeEntity> edges) {
    Map<String, KnowledgeGraphEdgeEntity> deduped = new LinkedHashMap<>();
    for (KnowledgeGraphEdgeEntity edge : edges) {
      String key = edge.getDomain() + "|" + edge.getSourceName() + "|" + edge.getRelation()
          + "|" + edge.getTargetName();
      deduped.putIfAbsent(key, edge);
    }
    return List.copyOf(deduped.values());
  }

  private GraphNodeDTO toNodeDTO(KnowledgeGraphNodeEntity entity) {
    return new GraphNodeDTO(
        nodeId(entity.getDomain(), entity.getName()),
        entity.getName(),
        entity.getType(),
        entity.getDomain(),
        entity.getMentionCount(),
        entity.getSourceFile()
    );
  }

  private GraphEdgeDTO toEdgeDTO(KnowledgeGraphEdgeEntity entity) {
    return new GraphEdgeDTO(
        entity.getId() == null
            ? entity.getDomain() + ":" + entity.getSourceName() + ":" + entity.getTargetName()
            : entity.getId().toString(),
        nodeId(entity.getDomain(), entity.getSourceName()),
        nodeId(entity.getDomain(), entity.getTargetName()),
        entity.getRelation(),
        entity.getConfidence(),
        entity.getEvidence(),
        entity.getSourceFile(),
        entity.getSectionTitle()
    );
  }

  private String nodeId(String domain, String name) {
    return domain + ":" + name;
  }

  private int clampLimit(Integer limit) {
    if (limit == null) {
      return DEFAULT_LIMIT;
    }
    return Math.max(1, Math.min(limit, MAX_LIMIT));
  }

  private String normalize(String value) {
    return isBlank(value) ? null : value.trim();
  }

  private boolean isBlank(String value) {
    return value == null || value.isBlank();
  }
}
