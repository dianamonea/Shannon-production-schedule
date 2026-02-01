"use client";

import React, { ReactNode, ReactElement } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  children: ReactNode;
  fallback?: (error: Error, reset: () => void) => ReactElement;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error caught by boundary:", error, errorInfo);
  }

  reset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset);
      }

      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-background">
          <Card className="max-w-md w-full border-destructive/50">
            <CardHeader>
              <div className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                <CardTitle>出错了</CardTitle>
              </div>
              <CardDescription>应用程序遇到了意外错误</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-3 bg-destructive/10 rounded-lg text-sm">
                <p className="font-mono text-destructive line-clamp-3">
                  {this.state.error.message}
                </p>
              </div>
              <div className="flex gap-2">
                <Button onClick={this.reset} className="flex-1" variant="default">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  重试
                </Button>
                <Button
                  onClick={() => window.location.href = "/"}
                  variant="outline"
                  className="flex-1"
                >
                  返回首页
                </Button>
              </div>
              {process.env.NODE_ENV === "development" && (
                <details className="text-xs text-muted-foreground">
                  <summary className="cursor-pointer font-semibold">详细信息</summary>
                  <pre className="mt-2 p-2 bg-muted rounded overflow-auto max-h-40">
                    {this.state.error.stack}
                  </pre>
                </details>
              )}
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}
