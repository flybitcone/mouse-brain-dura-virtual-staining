# 上传到 GitHub

1. 在 GitHub 创建一个空的 Public 仓库，建议名 `mouse-brain-dura-virtual-staining`。
2. 不勾选自动生成 README、LICENSE 或 .gitignore，本包已包含这些文件。
3. 将本文件夹的内容上传到仓库根目录，不要只上传 ZIP。
4. 上传后创建一个 Release，标签填 `v1.0.0`。
5. 将仓库网址发回，用于替换论文 Code Availability 的待补链接。

推荐在此文件夹打开 PowerShell，用 Git 上传：

```powershell
git init
git add .
git commit -m "Release preprocessing and benchmark code v1.0.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

把示例网址替换成刚创建的真实仓库网址。GitHub 登录使用本机 Git 凭据管理器。
也可网页分批上传；保留目录结构及隐藏的 `.gitignore` 文件。

不上传原始图像、NPY 数据、模型权重或访问令牌。数据放在 Zenodo。
代码为 MIT 许可，数据仍为 CC BY 4.0；第三方模型保留原许可证。

本包已检查 Python 语法和 ZIP 完整性。预处理单元测试不需要完整数据；
实际数据集的划分测试需要下载原始 TIFF。训练依赖 PyTorch，本次未重跑训练。
