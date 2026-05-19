package interview.guide.modules.knowledgegraph;

import interview.guide.common.annotation.RateLimit;
import interview.guide.common.result.Result;
import interview.guide.modules.knowledgegraph.model.GraphImportResponse;
import interview.guide.modules.knowledgegraph.model.GraphNodeDTO;
import interview.guide.modules.knowledgegraph.model.GraphNodeDetailDTO;
import interview.guide.modules.knowledgegraph.model.GraphResponse;
import interview.guide.modules.knowledgegraph.model.GraphStatsDTO;
import interview.guide.modules.knowledgegraph.service.KnowledgeGraphImportService;
import interview.guide.modules.knowledgegraph.service.KnowledgeGraphQueryService;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@Tag(name = "知识图谱管理", description = "知识图谱导入、查询、搜索与统计")
public class KnowledgeGraphController {

  private final KnowledgeGraphImportService importService;
  private final KnowledgeGraphQueryService queryService;

  @PostMapping("/api/knowledgegraph/import")
  @RateLimit(dimension = RateLimit.Dimension.GLOBAL, count = 1)
  @RateLimit(dimension = RateLimit.Dimension.IP, count = 1)
  public Result<GraphImportResponse> importKnowledgeGraph() {
    return Result.success(importService.importFromConfiguredPath());
  }

  @GetMapping("/api/knowledgegraph/graph")
  public Result<GraphResponse> getGraph(
      @RequestParam(value = "domain", required = false) String domain,
      @RequestParam(value = "keyword", required = false) String keyword,
      @RequestParam(value = "depth", required = false) Integer depth,
      @RequestParam(value = "limit", required = false) Integer limit
  ) {
    return Result.success(queryService.getGraph(domain, keyword, depth, limit));
  }

  @GetMapping("/api/knowledgegraph/nodes/search")
  public Result<List<GraphNodeDTO>> searchNodes(
      @RequestParam(value = "domain", required = false) String domain,
      @RequestParam("keyword") String keyword,
      @RequestParam(value = "limit", required = false) Integer limit
  ) {
    return Result.success(queryService.searchNodes(domain, keyword, limit));
  }

  @GetMapping("/api/knowledgegraph/nodes/{nodeName}")
  public Result<GraphNodeDetailDTO> getNodeDetail(
      @PathVariable String nodeName,
      @RequestParam("domain") String domain
  ) {
    return Result.success(queryService.getNodeDetail(domain, nodeName));
  }

  @GetMapping("/api/knowledgegraph/nodes/detail")
  public Result<GraphNodeDetailDTO> getNodeDetailByQuery(
      @RequestParam("domain") String domain,
      @RequestParam("nodeName") String nodeName
  ) {
    return Result.success(queryService.getNodeDetail(domain, nodeName));
  }

  @GetMapping("/api/knowledgegraph/stats")
  public Result<GraphStatsDTO> getStats() {
    return Result.success(queryService.getStats());
  }
}
