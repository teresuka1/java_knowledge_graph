package interview.guide.modules.knowledgegraph.service;

import interview.guide.common.exception.BusinessException;
import interview.guide.common.exception.ErrorCode;
import interview.guide.modules.knowledgegraph.model.GraphImportResponse;
import interview.guide.modules.knowledgegraph.model.KnowledgeGraphEdgeEntity;
import interview.guide.modules.knowledgegraph.model.KnowledgeGraphNodeEntity;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphEdgeRepository;
import interview.guide.modules.knowledgegraph.repository.KnowledgeGraphNodeRepository;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@RequiredArgsConstructor
public class KnowledgeGraphImportService {

  private static final String ENTITY_DIR = "实体抽取结果";
  private static final String RELATION_FILE = "关系抽取结果/all_relations.csv";

  private final KnowledgeGraphProperties properties;
  private final KnowledgeGraphNodeRepository nodeRepository;
  private final KnowledgeGraphEdgeRepository edgeRepository;
  private final KnowledgeGraphCsvParser csvParser = new KnowledgeGraphCsvParser();

  @Transactional
  public GraphImportResponse importFromConfiguredPath() {
    Path rootPath = resolveRootPath();
    log.info("Importing knowledge graph: rootPath={}", rootPath);

    try {
      Map<String, KnowledgeGraphNodeEntity> nodes = new HashMap<>();
      int skippedCount = importNodes(rootPath.resolve(ENTITY_DIR), nodes);
      EdgeImportResult edgeImportResult = importEdges(rootPath.resolve(RELATION_FILE), nodes);
      skippedCount += edgeImportResult.skippedCount();

      edgeRepository.deleteAllInBatch();
      nodeRepository.deleteAllInBatch();
      nodeRepository.saveAll(nodes.values());
      edgeRepository.saveAll(edgeImportResult.edges());

      log.info(
          "Knowledge graph imported: nodeCount={}, edgeCount={}, skippedCount={}",
          nodes.size(), edgeImportResult.edges().size(), skippedCount
      );
      return new GraphImportResponse(nodes.size(), edgeImportResult.edges().size(), skippedCount);
    } catch (IOException e) {
      throw new BusinessException(ErrorCode.KNOWLEDGE_GRAPH_IMPORT_FAILED, "读取知识图谱 CSV 失败", e);
    }
  }

  private int importNodes(Path entityDir, Map<String, KnowledgeGraphNodeEntity> nodes)
      throws IOException {
    if (!Files.isDirectory(entityDir)) {
      throw new BusinessException(
          ErrorCode.KNOWLEDGE_GRAPH_IMPORT_FAILED,
          "实体抽取结果目录不存在: " + entityDir
      );
    }

    int skippedCount = 0;
    try (var files = Files.list(entityDir)) {
      for (Path file : files.filter(path -> path.toString().endsWith(".csv")).toList()) {
        String domain = filenameWithoutExtension(file.getFileName().toString());
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        for (int i = 1; i < lines.size(); i++) {
          List<String> row = csvParser.parseLine(lines.get(i));
          if (row.size() < 5 || isBlank(row.get(0))) {
            skippedCount++;
            continue;
          }
          String name = row.get(0);
          nodes.put(nodeKey(domain, name), KnowledgeGraphNodeEntity.builder()
              .name(name)
              .type(row.get(1))
              .mentionCount(parseInteger(row.get(2)))
              .mentions(row.get(3))
              .sourceFile(row.get(4))
              .domain(domain)
              .build());
        }
      }
    }
    return skippedCount;
  }

  private EdgeImportResult importEdges(
      Path relationFile,
      Map<String, KnowledgeGraphNodeEntity> nodes
  ) throws IOException {
    if (!Files.isRegularFile(relationFile)) {
      throw new BusinessException(
          ErrorCode.KNOWLEDGE_GRAPH_IMPORT_FAILED,
          "关系抽取结果文件不存在: " + relationFile
      );
    }

    List<KnowledgeGraphEdgeEntity> edges = new ArrayList<>();
    int skippedCount = 0;
    List<String> lines = Files.readAllLines(relationFile, StandardCharsets.UTF_8);
    for (int i = 1; i < lines.size(); i++) {
      List<String> row = csvParser.parseLine(lines.get(i));
      if (row.size() < 11 || isBlank(row.get(0)) || isBlank(row.get(2))) {
        skippedCount++;
        continue;
      }

      String sourceFile = row.get(8);
      String domain = domainFromSourceFile(sourceFile);
      String sourceName = row.get(0);
      String targetName = row.get(2);
      nodes.putIfAbsent(
          nodeKey(domain, sourceName),
          fallbackNode(domain, sourceName, row.get(3), sourceFile)
      );
      nodes.putIfAbsent(
          nodeKey(domain, targetName),
          fallbackNode(domain, targetName, row.get(4), sourceFile)
      );

      edges.add(KnowledgeGraphEdgeEntity.builder()
          .sourceName(sourceName)
          .relation(row.get(1))
          .targetName(targetName)
          .sourceType(row.get(3))
          .targetType(row.get(4))
          .evidence(row.get(5))
          .patternName(row.get(6))
          .sectionTitle(row.get(7))
          .sourceFile(sourceFile)
          .domain(domain)
          .confidence(parseDouble(row.get(9)))
          .method(row.get(10))
          .build());
    }
    return new EdgeImportResult(edges, skippedCount);
  }

  private Path resolveRootPath() {
    List<Path> candidates = new ArrayList<>();
    if (properties.getRootPath() != null && !properties.getRootPath().isBlank()) {
      candidates.add(Path.of(properties.getRootPath()));
    }
    Path userDir = Path.of(System.getProperty("user.dir"));
    candidates.add(userDir.resolve("知识图谱"));
    candidates.add(userDir.resolve("../知识图谱").normalize());

    return candidates.stream()
        .map(Path::toAbsolutePath)
        .filter(Files::isDirectory)
        .findFirst()
        .orElseThrow(() -> new BusinessException(
            ErrorCode.KNOWLEDGE_GRAPH_IMPORT_FAILED,
            "未找到知识图谱目录，请检查 app.knowledge-graph.root-path 配置"
        ));
  }

  private KnowledgeGraphNodeEntity fallbackNode(
      String domain,
      String name,
      String type,
      String sourceFile
  ) {
    return KnowledgeGraphNodeEntity.builder()
        .name(name)
        .type(type)
        .mentionCount(0)
        .mentions(name)
        .sourceFile(sourceFile)
        .domain(domain)
        .build();
  }

  private String nodeKey(String domain, String name) {
    return domain + "\u0000" + name;
  }

  private String domainFromSourceFile(String sourceFile) {
    if (isBlank(sourceFile)) {
      return "default";
    }
    return filenameWithoutExtension(sourceFile);
  }

  private String filenameWithoutExtension(String filename) {
    int dotIndex = filename.lastIndexOf('.');
    return dotIndex > 0 ? filename.substring(0, dotIndex) : filename;
  }

  private Integer parseInteger(String value) {
    try {
      return Integer.parseInt(value);
    } catch (NumberFormatException e) {
      return 0;
    }
  }

  private Double parseDouble(String value) {
    try {
      return Double.parseDouble(value);
    } catch (NumberFormatException e) {
      return null;
    }
  }

  private boolean isBlank(String value) {
    return value == null || value.isBlank();
  }

  private record EdgeImportResult(
      List<KnowledgeGraphEdgeEntity> edges,
      int skippedCount
  ) {
  }
}
