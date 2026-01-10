# Persons 删除接口优化说明

## 📌 优化概述

针对 `/persons` 删除相关接口存在的问题进行了以下优化：

1. ✅ `DELETE /persons/by_name` 增加模糊删除风险提示
2. ✅ `DELETE /persons/by_id` 改为请求体传参，避免路由冲突
3. ✅ `DELETE /persons/delete` 作为通用删除入口（支持 `id` 或 `name`）
4. ✅ 统一使用全局 `db` 实例

---

## 🔧 具体优化内容

### **优化 1：`DELETE /persons/by_name` - 添加同名警告**

**风险说明**:
- 使用 MongoDB 正则模糊匹配
- 可能误删同名或包含关键字的人物

**建议**:
- 生产环境推荐使用 `DELETE /persons/by_id`
- 开发/测试环境批量清理可用此接口

---

### **优化 2：`DELETE /persons/by_id` - 请求体传参**

**问题**:
- 原路径参数 `/{person_id}` 易与其他路由冲突

**当前实现**:
```python
@router.delete("/by_id")
async def delete_person_by_id_api(
    request: DeletePersonByIdRequest = Body(...)
):
    info = await person_crud.delete_person(db, person_id=request.id)
```

**请求示例**:
```json
{
  "id": "507f1f77bcf86cd799439011"
}
```

---

### **优化 3：`DELETE /persons/delete` - 通用删除入口**

**说明**:
- 支持通过 `name` 或 `id` 删除
- 同时传入 `name` 和 `id` 时，**优先按 `name` 删除**
- `number` 字段目前不参与删除条件（预留字段）

**请求示例**:
```json
{ "name": "张三" }
```
或
```json
{ "id": "507f1f77bcf86cd799439011" }
```

**响应示例**:
- 按 `name` 删除：
```json
{ "deleted_count": 2, "message": "成功删除 2 个人物" }
```
- 按 `id` 删除：
```json
{ "message": "人物已删除", "id": "507f1f77bcf86cd799439011" }
```

---

## 📊 接口对比总结

| 接口 | 优化前 | 优化后 | 说明 |
|------|--------|--------|------|
| **DELETE /persons/by_name** | 模糊匹配，无提示 | 添加风险提示 | 保留向下兼容，但不推荐生产使用 |
| **DELETE /persons/{person_id}** | 路径参数 | 改为 `/persons/by_id` + 请求体 | 避免路由冲突，风格统一 |
| **DELETE /persons/delete** | 不存在 | 通用入口 | 支持 `id`/`name` 删除 |

---

## 🎯 推荐的删除接口使用指南

### **场景 1：精确删除单个人物（推荐）**

**使用接口**: `DELETE /persons/by_id`

```json
{ "id": "507f1f77bcf86cd799439011" }
```

---

### **场景 2：按条件删除（推荐）**

**使用接口**: `DELETE /persons/delete`

```json
{ "name": "张三" }
```
或
```json
{ "id": "507f1f77bcf86cd799439011" }
```

---

### **场景 3：批量模糊删除（不推荐）**

**使用接口**: `DELETE /persons/by_name?name=张三`

**风险**:
- ⚠️ 模糊匹配，可能误删
- ⚠️ 仅建议开发/测试环境使用

---

## 🔄 迁移建议

**旧调用**:
```javascript
await fetch(`/persons/${personId}`, { method: 'DELETE' });
```

**新调用**:
```javascript
await fetch('/persons/by_id', {
  method: 'DELETE',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ id: personId })
});
```

---

## 🛡️ 安全建议

1. **生产环境**
   - ✅ 仅使用 `DELETE /persons/by_id` 或 `DELETE /persons/delete`（带 `id`）
   - ❌ 禁用 `DELETE /persons/by_name`

2. **开发环境**
   - ✅ 可使用 `DELETE /persons/by_name` 快速清理测试数据

3. **最佳实践**
   - 删除前先查询确认（使用 `POST /persons/search`）
   - 重要操作添加二次确认
   - 记录删除日志

---

## 📝 更新日志

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v2.1 | 2026-01-09 | 补充通用删除接口说明与响应示例 |
| v2.0 | 2026-01-07 | 优化删除接口，添加安全警告，统一 db 使用 |
| v1.0 | - | 初始版本 |

---

## 📞 技术支持

如有疑问，请参考：
- [RECOGNIZE_API_ERRORS.md](./RECOGNIZE_API_ERRORS.md) - `/recognize` 接口错误码说明
- 后端日志中的 `[/persons/*]` 相关日志
