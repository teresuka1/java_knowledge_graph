package interview.guide.modules.knowledgegraph.service;

import interview.guide.modules.knowledgegraph.model.KnowledgeGraphEdgeEntity;
import interview.guide.modules.knowledgegraph.model.KnowledgeGraphNodeEntity;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphEdgeRepository;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphNodeRepository;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;

@DisplayName("知识图谱导入服务测试")
@SuppressWarnings({"rawtypes", "unchecked"})
class KnowledgeGraphImportServiceTest {

  private KnowledgeGraphImportService importService;

  @Mock
  private KnowledgeGraphNodeRepository nodeRepository;

  @Mock
  private KnowledgeGraphEdgeRepository edgeRepository;

  @TempDir
  private Path tempDir;

  @BeforeEach
  void setUp() {
    MockitoAnnotations.openMocks(this);
    KnowledgeGraphProperties properties = new KnowledgeGraphProperties();
    properties.setRootPath(tempDir.toString());
    importService = new KnowledgeGraphImportService(properties, nodeRepository, edgeRepository);
  }

  @Nested
  @DisplayName("CSV 导入")
  class CsvImportTests {

    @Test
    @DisplayName("应从实体和关系 CSV 导入节点与关系")
    void shouldImportNodesAndEdgesFromCsv() throws IOException {
      writeCsvFiles();

      var result = importService.importFromConfiguredPath();

      ArgumentCaptor<Iterable<KnowledgeGraphNodeEntity>> nodeCaptor =
          (ArgumentCaptor) ArgumentCaptor.forClass(Iterable.class);
      ArgumentCaptor<Iterable<KnowledgeGraphEdgeEntity>> edgeCaptor =
          (ArgumentCaptor) ArgumentCaptor.forClass(Iterable.class);
      verify(nodeRepository).saveAll(nodeCaptor.capture());
      verify(edgeRepository).saveAll(edgeCaptor.capture());

      List<KnowledgeGraphNodeEntity> nodes = toList(nodeCaptor.getValue());
      List<KnowledgeGraphEdgeEntity> edges = toList(edgeCaptor.getValue());

      assertThat(result.nodeCount()).isEqualTo(3);
      assertThat(result.edgeCount()).isEqualTo(1);
      assertThat(nodes).extracting(KnowledgeGraphNodeEntity::getName)
          .containsExactlyInAnyOrder("Java", "JVM", "Heap");
      assertThat(edges).singleElement().satisfies(edge -> {
        assertThat(edge.getSourceName()).isEqualTo("JVM");
        assertThat(edge.getRelation()).isEqualTo("has_part");
        assertThat(edge.getTargetName()).isEqualTo("Heap");
        assertThat(edge.getEvidence()).isEqualTo("JVM contains heap, stack and runtime areas");
      });
    }
  }

  private void writeCsvFiles() throws IOException {
    Path entityDir = tempDir.resolve("实体抽取结果");
    Path relationDir = tempDir.resolve("关系抽取结果");
    Files.createDirectories(entityDir);
    Files.createDirectories(relationDir);

    Files.writeString(
        entityDir.resolve("java.csv"),
        """
        main_entity,entity_type,mention_count,mentions,source_file
        Java,领域实体,3,Java,java.txt
        JVM,领域实体,5,JVM,java.txt
        """,
        StandardCharsets.UTF_8
    );

    Files.writeString(
        relationDir.resolve("all_relations.csv"),
        "head,relation,tail,head_type,tail_type,evidence,pattern_name,"
            + "section_title,source_file,confidence,method\n"
            + "JVM,has_part,Heap,Domain,Memory,"
            + "\"JVM contains heap, stack and runtime areas\","
            + "Part,JVM memory,java.txt,0.91,rule\n",
        StandardCharsets.UTF_8
    );
  }

  private <T> List<T> toList(Iterable<T> iterable) {
    List<T> values = new ArrayList<>();
    iterable.forEach(values::add);
    return values;
  }
}
