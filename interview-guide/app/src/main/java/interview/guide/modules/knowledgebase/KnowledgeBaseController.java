package interview.guide.modules.knowledgebase;

import interview.guide.common.annotation.RateLimit;
import interview.guide.common.result.Result;
import interview.guide.modules.knowledgebase.model.KnowledgeBaseListItemDTO;
import interview.guide.modules.knowledgebase.model.KnowledgeBaseStatsDTO;
import interview.guide.modules.knowledgebase.model.VectorStatus;
import interview.guide.modules.knowledgebase.service.KnowledgeBaseDeleteService;
import interview.guide.modules.knowledgebase.service.KnowledgeBaseListService;
import interview.guide.modules.knowledgebase.service.KnowledgeBaseUploadService;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequiredArgsConstructor
@Tag(name = "知识库管理", description = "知识库上传、下载、分类与向量化")
public class KnowledgeBaseController {

  private final KnowledgeBaseUploadService uploadService;
  private final KnowledgeBaseListService listService;
  private final KnowledgeBaseDeleteService deleteService;

  @GetMapping("/api/knowledgebase/list")
  public Result<List<KnowledgeBaseListItemDTO>> getAllKnowledgeBases(
      @RequestParam(value = "sortBy", required = false) String sortBy,
      @RequestParam(value = "vectorStatus", required = false) String vectorStatus) {
    VectorStatus status = null;
    if (vectorStatus != null && !vectorStatus.isBlank()) {
      status = VectorStatus.valueOf(vectorStatus.toUpperCase());
    }
    return Result.success(listService.listKnowledgeBases(status, sortBy));
  }

  @GetMapping("/api/knowledgebase/{id}")
  public Result<KnowledgeBaseListItemDTO> getKnowledgeBase(@PathVariable Long id) {
    return listService.getKnowledgeBase(id)
        .map(Result::success)
        .orElse(Result.error("知识库不存在"));
  }

  @DeleteMapping("/api/knowledgebase/{id}")
  public Result<Void> deleteKnowledgeBase(@PathVariable Long id) {
    deleteService.deleteKnowledgeBase(id);
    return Result.success(null);
  }

  @GetMapping("/api/knowledgebase/categories")
  public Result<List<String>> getAllCategories() {
    return Result.success(listService.getAllCategories());
  }

  @GetMapping("/api/knowledgebase/category/{category}")
  public Result<List<KnowledgeBaseListItemDTO>> getByCategory(@PathVariable String category) {
    return Result.success(listService.listByCategory(category));
  }

  @GetMapping("/api/knowledgebase/uncategorized")
  public Result<List<KnowledgeBaseListItemDTO>> getUncategorized() {
    return Result.success(listService.listByCategory(null));
  }

  @PutMapping("/api/knowledgebase/{id}/category")
  public Result<Void> updateCategory(@PathVariable Long id, @RequestBody Map<String, String> body) {
    listService.updateCategory(id, body.get("category"));
    return Result.success(null);
  }

  @PostMapping(value = "/api/knowledgebase/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
  @RateLimit(dimension = RateLimit.Dimension.GLOBAL, count = 3)
  @RateLimit(dimension = RateLimit.Dimension.IP, count = 3)
  public Result<Map<String, Object>> uploadKnowledgeBase(
      @RequestParam("file") MultipartFile file,
      @RequestParam(value = "name", required = false) String name,
      @RequestParam(value = "category", required = false) String category) {
    return Result.success(uploadService.uploadKnowledgeBase(file, name, category));
  }

  @GetMapping("/api/knowledgebase/{id}/download")
  public ResponseEntity<byte[]> downloadKnowledgeBase(@PathVariable Long id) {
    var entity = listService.getEntityForDownload(id);
    byte[] fileContent = listService.downloadFile(id);
    String encodedFilename = URLEncoder.encode(entity.getOriginalFilename(), StandardCharsets.UTF_8)
        .replaceAll("\\+", "%20");
    return ResponseEntity.ok()
        .header(
            HttpHeaders.CONTENT_DISPOSITION,
            "attachment; filename=\"" + encodedFilename + "\"; filename*=UTF-8''" + encodedFilename
        )
        .header(
            HttpHeaders.CONTENT_TYPE,
            entity.getContentType() != null ? entity.getContentType() : MediaType.APPLICATION_OCTET_STREAM_VALUE
        )
        .body(fileContent);
  }

  @GetMapping("/api/knowledgebase/search")
  public Result<List<KnowledgeBaseListItemDTO>> search(@RequestParam("keyword") String keyword) {
    return Result.success(listService.search(keyword));
  }

  @GetMapping("/api/knowledgebase/stats")
  public Result<KnowledgeBaseStatsDTO> getStatistics() {
    return Result.success(listService.getStatistics());
  }

  @PostMapping("/api/knowledgebase/{id}/revectorize")
  @RateLimit(dimension = RateLimit.Dimension.GLOBAL, count = 2)
  @RateLimit(dimension = RateLimit.Dimension.IP, count = 2)
  public Result<Void> revectorize(@PathVariable Long id) {
    uploadService.revectorize(id);
    return Result.success(null);
  }
}
