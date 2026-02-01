"use client";

export function LoadingSkeleton() {
  return (
    <div className="space-y-4 p-4">
      {/* 标题骨架 */}
      <div className="space-y-2">
        <div className="h-8 bg-muted rounded-lg w-2/5 animate-pulse" />
        <div className="h-4 bg-muted rounded-lg w-3/5 animate-pulse" />
      </div>

      {/* 按钮组骨架 */}
      <div className="flex gap-2">
        <div className="h-10 bg-muted rounded-lg w-24 animate-pulse" />
        <div className="h-10 bg-muted rounded-lg w-32 animate-pulse" />
      </div>

      {/* 卡片网格骨架 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="space-y-3 p-4 border rounded-lg">
            <div className="flex items-center gap-2">
              <div className="h-10 w-10 bg-muted rounded-lg animate-pulse" />
              <div className="flex-1">
                <div className="h-4 bg-muted rounded w-3/4 animate-pulse mb-2" />
                <div className="h-3 bg-muted rounded w-1/2 animate-pulse" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="h-3 bg-muted rounded w-full animate-pulse" />
              <div className="h-3 bg-muted rounded w-4/5 animate-pulse" />
            </div>
          </div>
        ))}
      </div>

      {/* 内容卡片骨架 */}
      <div className="space-y-2">
        <div className="h-6 bg-muted rounded-lg w-40 animate-pulse" />
        <div className="p-4 border rounded-lg space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="space-y-2">
              <div className="flex gap-2">
                <div className="h-3 w-3 bg-muted rounded-full animate-pulse mt-1" />
                <div className="flex-1">
                  <div className="h-4 bg-muted rounded w-3/4 animate-pulse mb-2" />
                  <div className="h-3 bg-muted rounded w-full animate-pulse" />
                  <div className="h-3 bg-muted rounded w-5/6 animate-pulse mt-1" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function PageLoadingSpinner() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-background/50 backdrop-blur-sm z-50">
      <div className="space-y-4 text-center">
        <div className="relative w-16 h-16 mx-auto">
          <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary animate-spin" />
        </div>
        <p className="text-lg font-semibold text-foreground">加载中...</p>
      </div>
    </div>
  );
}

export function DataEmptyState({
  title = "暂无数据",
  description = "当前没有任何数据可显示",
  icon: Icon,
  action,
}: {
  title?: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4">
      {Icon && <Icon className="h-16 w-16 text-muted-foreground mb-4 opacity-50" />}
      <h3 className="text-lg font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground mb-6 max-w-sm text-center">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
}
