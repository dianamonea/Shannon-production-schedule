"use client";

import { AppSidebar } from "@/components/app-sidebar";
import { AuthGuard } from "@/components/auth-guard";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";

export function AppLayout({ children }: { children: React.ReactNode }) {
    return (
        <AuthGuard>
            <SidebarProvider>
                <div className="flex h-screen w-screen overflow-hidden bg-background">
                    <AppSidebar />
                    <main className="flex-1 flex flex-col overflow-hidden">
                        {/* 顶部工具栏 */}
                        <div className="flex items-center justify-between border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4 py-2 shrink-0 sticky top-0 z-10 h-14">
                            <div className="flex items-center gap-2">
                                <SidebarTrigger className="cursor-pointer" />
                            </div>
                        </div>
                        {/* 内容区域 */}
                        <div className="flex-1 min-h-0 overflow-hidden">
                            {children}
                        </div>
                    </main>
                </div>
            </SidebarProvider>
        </AuthGuard>
    );
}
