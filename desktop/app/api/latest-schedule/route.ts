import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
    try {
        // 获取 Shannon 项目根目录
        const shannonRoot = path.join(process.cwd(), '..', '..', '..');
        
        // 查找最新的 schedule_result_*.json 文件
        const files = fs.readdirSync(shannonRoot)
            .filter(f => f.startsWith('schedule_result_') && f.endsWith('.json'))
            .map(f => ({
                name: f,
                path: path.join(shannonRoot, f),
                time: fs.statSync(path.join(shannonRoot, f)).mtime.getTime()
            }))
            .sort((a, b) => b.time - a.time);

        if (files.length === 0) {
            return NextResponse.json(
                { error: '未找到调度结果文件' },
                { status: 404 }
            );
        }

        // 读取最新的文件
        const latestFile = files[0];
        const content = fs.readFileSync(latestFile.path, 'utf-8');
        const data = JSON.parse(content);

        return NextResponse.json(data);
    } catch (error) {
        console.error('Failed to fetch latest schedule:', error);
        return NextResponse.json(
            { error: '无法读取调度结果', details: String(error) },
            { status: 500 }
        );
    }
}
