package interview.guide.modules.knowledgegraph.service;

import interview.guide.common.exception.BusinessException;
import interview.guide.modules.knowledgegraph.model.GraphNodeDTO;
import interview.guide.modules.knowledgegraph.model.KnowledgeGraphEdgeEntity;
import interview.guide.modules.knowledgegraph.model.KnowledgeGraphNodeEntity;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphEdgeRepository;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphNodeRepository;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.data.domain.Pageable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyCollection;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

@DisplayName("知识图谱查询服务测试")
class KnowledgeGraphQueryServiceTest {

  private KnowledgeGraphQueryService queryService;

  @Mock
  private KnowledgeGraphNodeRepository nodeRepository;

  @Mock
  private KnowledgeGraphEdgeRepository edgeRepository;

  @BeforeEach
  void setUp() {
    MockitoAnnotations.openMocks(this);
    queryService = new KnowledgeGraphQueryService(nodeRepository, edgeRepository);
  }

  @Nested
  @DisplayName("图谱查询")
  class GraphQueryTests {

    @Test
    @DisplayName("根据关键词查询时应返回种子节点和一度关系")
    void shouldReturnSeedNodesAndFirstLevelRelations() {
      KnowledgeGraphNodeEntity jvm = node("JVM", "领域实体");
      KnowledgeGraphNodeEntity heap = node("Heap", "内存实体");
      KnowledgeGraphEdgeEntity edge = edge("JVM", "has_part", "Heap");

      when(nodeRepository.searchNodes(eq("java"), eq("JVM"), any(Pageable.class)))
          .thenReturn(List.of(jvm));
      when(edgeRepository.findEdgesTouching(eq("java"), anyCollection(), any(Pageable.class)))
          .thenReturn(List.of(edge));
      when(nodeRepository.findByDomainAndNameIn(eq("java"), anyCollection()))
          .thenReturn(List.of(jvm, heap));

      var graph = queryService.getGraph("java", "JVM", 1, 100);

      assertThat(graph.nodes())
          .extracting(GraphNodeDTO::name)
          .containsExactlyInAnyOrder("JVM", "Heap");
      assertThat(graph.edges()).singleElement().satisfies(resultEdge -> {
        assertThat(resultEdge.source()).isEqualTo("java:JVM");
        assertThat(resultEdge.target()).isEqualTo("java:Heap");
        assertThat(resultEdge.relation()).isEqualTo("has_part");
      });
    }
  }

  @Nested
  @DisplayName("节点详情")
  class NodeDetailTests {

    @Test
    @DisplayName("节点不存在时应抛出业务异常")
    void shouldThrowBusinessExceptionWhenNodeMissing() {
      when(nodeRepository.findByDomainAndName("java", "不存在")).thenReturn(Optional.empty());

      assertThatThrownBy(() -> queryService.getNodeDetail("java", "不存在"))
          .isInstanceOf(BusinessException.class)
          .hasMessageContaining("知识图谱节点不存在");
    }
  }

  private KnowledgeGraphNodeEntity node(String name, String type) {
    return KnowledgeGraphNodeEntity.builder()
        .name(name)
        .type(type)
        .domain("java")
        .mentionCount(3)
        .sourceFile("java.txt")
        .build();
  }

  private KnowledgeGraphEdgeEntity edge(String source, String relation, String target) {
    return KnowledgeGraphEdgeEntity.builder()
        .id(1L)
        .sourceName(source)
        .relation(relation)
        .targetName(target)
        .domain("java")
        .confidence(0.91)
        .evidence("JVM 包含堆")
        .sourceFile("java.txt")
        .build();
  }
}
