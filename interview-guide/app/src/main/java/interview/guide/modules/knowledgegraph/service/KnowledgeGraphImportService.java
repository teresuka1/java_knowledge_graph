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
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
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

  private static final String DISAMBIGUATED_ENTITY_DIR = "实体消歧/实体消歧结果";
  private static final String LEGACY_ENTITY_DIR = "实体抽取结果";
  private static final String DISAMBIGUATED_RELATION_FILE = "关系消歧/关系消歧结果/all_relations_disambiguated.csv";
  private static final String LEGACY_RELATION_FILE = "关系抽取结果/all_relations.csv";

  private final KnowledgeGraphProperties properties;
  private final KnowledgeGraphNodeRepository nodeRepository;
  private final KnowledgeGraphEdgeRepository edgeRepository;
  private final KnowledgeGraphCsvParser csvParser = new KnowledgeGraphCsvParser();

  @Transactional
  public GraphImportResponse importFromConfiguredPath() {
    Path rootPath = resolveRootPath();
    log.info("Importing knowledge graph: rootPath={}", rootPath);

    try {
      Path entityDir = resolveEntityDir(rootPath);
      Path relationFile = resolveRelationFile(rootPath);

      Map<String, KnowledgeGraphNodeEntity> nodes = new HashMap<>();
      int skippedCount = importNodes(entityDir, nodes);
      EdgeImportResult edgeImportResult = importEdges(relationFile, nodes);
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
          "实体结果目录不存在: " + entityDir
      );
    }

    int skippedCount = 0;
    try (var files = Files.list(entityDir)) {
      for (Path file : files.filter(path -> path.toString().endsWith(".csv")).toList()) {
        String domain = filenameWithoutExtension(file.getFileName().toString());
        for (Map<String, String> row : readCsvRows(file)) {
          String name = firstNonBlank(row, "main_entity", "entity", "name");
          if (isBlank(name)) {
            skippedCount++;
            continue;
          }
          String type = firstNonBlank(row, "entity_type", "type");
          String mentions = firstNonBlank(row, "mentions", "aliases", "merged_entities");
          String sourceFile = firstNonBlank(row, "source_file");
          nodes.put(nodeKey(domain, name), KnowledgeGraphNodeEntity.builder()
              .name(name)
              .type(type)
              .mentionCount(parseInteger(firstNonBlank(row, "mention_count", "mentionCount")))
              .mentions(isBlank(mentions) ? name : mentions)
              .sourceFile(sourceFile)
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
          "关系结果文件不存在: " + relationFile
      );
    }

    List<KnowledgeGraphEdgeEntity> edges = new ArrayList<>();
    int skippedCount = 0;
    for (Map<String, String> row : readCsvRows(relationFile)) {
      String sourceName = firstNonBlank(row, "head", "sourceName", "source");
      String targetName = firstNonBlank(row, "tail", "targetName", "target");
      String relation = firstNonBlank(row, "relation");
      if (isBlank(sourceName) || isBlank(targetName) || isBlank(relation)) {
        skippedCount++;
        continue;
      }

      String sourceFile = pickFirstValue(firstNonBlank(row, "source_file", "source_files"));
      String domain = domainFromSourceFile(sourceFile);
      String sourceType = firstNonBlank(row, "head_type", "source_type", "sourceType");
      String targetType = firstNonBlank(row, "tail_type", "target_type", "targetType");

      nodes.putIfAbsent(
          nodeKey(domain, sourceName),
          fallbackNode(domain, sourceName, sourceType, sourceFile)
      );
      nodes.putIfAbsent(
          nodeKey(domain, targetName),
          fallbackNode(domain, targetName, targetType, sourceFile)
      );

      edges.add(KnowledgeGraphEdgeEntity.builder()
          .sourceName(sourceName)
          .relation(relation)
          .targetName(targetName)
          .sourceType(sourceType)
          .targetType(targetType)
          .evidence(firstNonBlank(row, "evidence"))
          .patternName(pickFirstValue(firstNonBlank(row, "pattern_name", "pattern_names")))
          .sectionTitle(pickFirstValue(firstNonBlank(row, "section_title", "section_titles")))
          .sourceFile(sourceFile)
          .domain(domain)
          .confidence(parseDouble(firstNonBlank(row, "confidence", "disambiguation_score")))
          .method(firstNonBlank(row, "method", "disambiguation_method"))
          .build());
    }
    return new EdgeImportResult(edges, skippedCount);
  }

  private Path resolveEntityDir(Path rootPath) {
    List<Path> candidates = List.of(
        rootPath.resolve(DISAMBIGUATED_ENTITY_DIR),
        rootPath.resolve(LEGACY_ENTITY_DIR)
    );
    return candidates.stream()
        .filter(Files::isDirectory)
        .findFirst()
        .orElseThrow(() -> new BusinessException(
            ErrorCode.KNOWLEDGE_GRAPH_IMPORT_FAILED,
            "未找到实体结果目录: " + candidates
        ));
  }

  private Path resolveRelationFile(Path rootPath) {
    List<Path> candidates = List.of(
        rootPath.resolve(DISAMBIGUATED_RELATION_FILE),
        rootPath.resolve(LEGACY_RELATION_FILE)
    );
    return candidates.stream()
        .filter(Files::isRegularFile)
        .findFirst()
        .orElseThrow(() -> new BusinessException(
            ErrorCode.KNOWLEDGE_GRAPH_IMPORT_FAILED,
            "未找到关系结果文件: " + candidates
        ));
  }

  private Path resolveRootPath() {
    List<Path> candidates = new ArrayList<>();
    if (properties.getRootPath() != null && !properties.getRootPath().isBlank()) {
      candidates.add(Path.of(properties.getRootPath()));
    }
    Path userDir = Path.of(System.getProperty("user.dir"));
    candidates.add(userDir);
    candidates.add(userDir.resolve("graph"));
    candidates.add(userDir.resolve("../graph").normalize());
    candidates.add(userDir.resolve("..").normalize());
    candidates.add(userDir.resolve("../..").normalize());

    return candidates.stream()
        .map(Path::toAbsolutePath)
        .filter(Files::isDirectory)
        .filter(this::containsKnowledgeGraphInputs)
        .findFirst()
        .orElseThrow(() -> new BusinessException(
            ErrorCode.KNOWLEDGE_GRAPH_IMPORT_FAILED,
            "未找到知识图谱目录，请检查 app.knowledge-graph.root-path 配置"
        ));
  }

  private boolean containsKnowledgeGraphInputs(Path rootPath) {
    return Files.isDirectory(rootPath.resolve(DISAMBIGUATED_ENTITY_DIR))
        || Files.isDirectory(rootPath.resolve(LEGACY_ENTITY_DIR))
        || Files.isRegularFile(rootPath.resolve(DISAMBIGUATED_RELATION_FILE))
        || Files.isRegularFile(rootPath.resolve(LEGACY_RELATION_FILE));
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

  private List<Map<String, String>> readCsvRows(Path file) throws IOException {
    List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
    if (lines.isEmpty()) {
      return List.of();
    }
    List<String> headers = csvParser.parseLine(lines.get(0)).stream()
        .map(String::trim)
        .toList();
    List<Map<String, String>> rows = new ArrayList<>();
    for (int i = 1; i < lines.size(); i++) {
      if (lines.get(i) == null || lines.get(i).isBlank()) {
        continue;
      }
      List<String> values = csvParser.parseLine(lines.get(i));
      Map<String, String> row = new LinkedHashMap<>();
      for (int col = 0; col < headers.size(); col++) {
        row.put(headers.get(col), col < values.size() ? values.get(col).trim() : "");
      }
      rows.add(row);
    }
    return rows;
  }

  private String firstNonBlank(Map<String, String> row, String... keys) {
    return Arrays.stream(keys)
        .map(row::get)
        .filter(value -> value != null && !value.isBlank())
        .map(String::trim)
        .findFirst()
        .orElse("");
  }

  private String pickFirstValue(String value) {
    if (isBlank(value)) {
      return value;
    }
    return Arrays.stream(value.split("\\s*\\|\\s*"))
        .map(String::trim)
        .filter(part -> !part.isBlank())
        .findFirst()
        .orElse(value.trim());
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
