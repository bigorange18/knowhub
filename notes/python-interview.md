### Window下删除不要的文件

```bash
Dism.exe /online /Cleanup-Image /StartComponentCleanup

Dism.exe /online /Cleanup-Image /StartComponentCleanup /ResetBase

Dism.exe /online /Cleanup-Image /SPSuperseded
```

