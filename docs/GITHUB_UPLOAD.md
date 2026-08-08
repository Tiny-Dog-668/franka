# 使用 GitHub 管理和上传项目

本文说明如何通过 SSH 将本地项目上传到 GitHub，以及后续如何提交更新。

## 1. 配置 Git 身份

```bash
git config --global user.name "你的名字"
git config --global user.email "你的GitHub邮箱"
```

## 2. 配置 GitHub SSH

已有 SSH 密钥时可跳过生成步骤。检查密钥：

```bash
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
```

没有时生成：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

将输出的公钥添加到 GitHub：

```text
Settings → SSH and GPG keys → New SSH key
```

测试连接：

```bash
ssh -T git@github.com
```

出现 `You've successfully authenticated` 表示配置成功。不要公开没有 `.pub` 后缀的私钥。

## 3. 在 GitHub 创建仓库

打开 <https://github.com/new>，填写 Owner、仓库名称和可见性。

本地已有项目时，不要初始化以下内容：

- README
- `.gitignore`
- License

创建完成后复制 SSH 地址，例如：

```text
git@github.com:用户名/仓库名.git
```

## 4. 初始化并提交本地项目

```bash
cd /home/td/franka
git init -b main                 # 已是 Git 仓库时跳过
git add .
git status
git diff --cached --stat
git commit -m "Initial commit"
```

提交前应通过 `.gitignore` 排除虚拟环境、模型权重、运行日志和其他大文件。

## 5. 绑定并推送远程仓库

没有 `origin` 时：

```bash
git remote add origin git@github.com:用户名/仓库名.git
```

如果 `origin` 已存在，更新地址：

```bash
git remote set-url origin git@github.com:用户名/仓库名.git
```

检查并首次推送：

```bash
git remote -v
git push -u origin main
```

## 6. 后续更新

```bash
git status
git add .
git diff --cached --stat
git commit -m "说明本次修改"
git push
```

## 常见问题

### `remote origin already exists`

```bash
git remote set-url origin git@github.com:用户名/仓库名.git
```

### `Repository not found`

确认 GitHub 仓库已经创建，并检查用户名和仓库名：

```bash
git remote -v
ssh -T git@github.com
```

### 模型或日志没有上传

本项目通过 `.gitignore` 排除了 `.pt` 模型、`runs/`、`.venv/` 和压缩包。这些文件应使用 Git LFS、GitHub Release 或其他存储方式单独管理。
