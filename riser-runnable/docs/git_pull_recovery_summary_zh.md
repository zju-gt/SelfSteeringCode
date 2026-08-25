# Git Pull 成功过程总结

## 1. 背景

2026-08-26 需要将本地仓库同步到 GitHub `origin/main` 的最新版本。

同步前的本地状态：

```text
HEAD: ade6e7a Add reasoning-first MMLU evaluation prompts
branch: main
working tree: clean
```

远端最新提交为：

```text
942df90 data_0825_2232
```

## 2. 遇到的问题

直接执行：

```bash
git pull origin main
```

命令长时间没有输出，也没有立即报告冲突。使用带低速超时的 pull 再次尝试后，仍然长时间等待，因此终止了卡住的进程。

随后先检查远端分支：

```bash
git ls-remote origin refs/heads/main
```

返回：

```text
942df90fca429ec0f8879d327214f3d5cffecaaa refs/heads/main
```

这说明 GitHub 网络可访问，并且远端确实存在更新；问题不是“远端没有新代码”。

进一步检查发现，新的 commit 对象已经部分下载到本地，但它对应的 tree 对象尚未完整可用：

```text
fatal: unable to read tree (...)
```

同时 Git 对象目录中存在中断传输留下的临时 pack 文件。此时不能直接 fast-forward，因为 commit 的文件树还不完整。

## 3. 最终采用的同步方法

### 3.1 补全远端对象

使用部分 clone 过滤，先拉取提交和目录结构，避免一次性传输所有大型 blob：

```bash
git -c http.version=HTTP/1.1 \
    fetch --no-tags --filter=blob:none origin main
```

成功输出：

```text
From https://github.com/zju-gt/SelfSteeringCode
 * branch            main       -> FETCH_HEAD
   ade6e7a..942df90  main       -> origin/main
```

其中：

- `--no-tags`：不下载本次同步不需要的 tag。
- `--filter=blob:none`：先获取 commit/tree，文件内容在 checkout 时按需获取。
- `http.version=HTTP/1.1`：在当前网络环境下提高 GitHub HTTP 传输的稳定性。

### 3.2 安全地 fast-forward

确认远端提交是当前本地提交的后继后，执行：

```bash
git merge --ff-only origin/main
```

`--ff-only` 要求只能直接前进，不自动创建 merge commit，也不会覆盖本地分支历史。

最终更新成功：

```text
Updating ade6e7a..942df90
Fast-forward
```

## 4. 成功原因

这次 pull 最终成功，关键原因有以下几点：

1. 先使用 `git ls-remote` 确认了远端真实 commit，避免把网络卡顿误判为“没有更新”。
2. 将 `pull` 拆成 `fetch` 和 `merge`，可以分别处理“对象下载”和“本地分支更新”两个阶段。
3. 使用 `--filter=blob:none` 降低了大型 artifacts 对传输过程的影响。
4. 本地工作区没有未提交修改，因此可以安全地执行 fast-forward。
5. 使用 `git merge --ff-only`，避免了不必要的 merge commit 和潜在冲突。
6. 最终 Git 自动按需补齐了 checkout 所需的文件内容。

当前仓库也记录了部分 clone 配置：

```text
remote.origin.promisor=true
remote.origin.partialclonefilter=blob:none
```

这表示后续 Git 可以继续按需下载大文件对象。

## 5. 最终验证

同步后执行：

```bash
git status --short --branch
git log -1 --oneline --decorate
```

结果为：

```text
## main...origin/main
942df90 (HEAD -> main, origin/main, origin/HEAD) data_0825_2232
```

这同时说明：

- 本地 `main` 已经指向 `942df90`。
- `origin/main` 已经同步到同一个 commit。
- 工作区没有未提交修改。

本次同步包含以下重要更新：

- 新的 MMLU prompt pairs 和 evaluation 数据。
- 更新后的 primitive library 和 vectors。
- 新结果文件：
  `artifacts/mmlu_preliminary/results_260825_1343.jsonl`
- 更新后的 `scripts/run_mmlu_preliminary.sh`。

## 6. 后续遇到类似问题时的推荐流程

```bash
# 1. 检查本地状态
git status --short --branch

# 2. 查询远端真实版本
git ls-remote origin refs/heads/main

# 3. 补全远端对象
git -c http.version=HTTP/1.1 \
    fetch --no-tags --filter=blob:none origin main

# 4. 只允许安全的 fast-forward
git merge --ff-only origin/main

# 5. 验证同步结果
git status --short --branch
git log -1 --oneline --decorate
```

不要在这种场景下直接使用 `git reset --hard` 或手动删除 Git 对象。应先确认远端 commit、补全对象，再执行 fast-forward。
