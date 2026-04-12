import { GraduationCap, ShieldCheck } from 'lucide-react'

export default function LoginPage({ onLogin }) {
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-primary-900 to-slate-900 flex items-center justify-center p-4">
            {/* Decorative blobs */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 -right-40 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl" />
                <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-primary-400/10 rounded-full blur-3xl" />
            </div>

            <div className="relative w-full max-w-md animate-fade-in">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 mb-4 shadow-lg shadow-primary-500/30">
                        <span className="text-white font-bold text-2xl">CE</span>
                    </div>
                    <h1 className="text-3xl font-bold text-white">ClassroomEye</h1>
                    <p className="text-slate-400 mt-2">AI-Powered Engagement Analytics</p>
                </div>

                {/* Login Card */}
                <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl">
                    <h2 className="text-xl font-semibold text-white text-center mb-2">Welcome Back</h2>
                    <p className="text-slate-400 text-center text-sm mb-8">Choose your role to continue</p>

                    <div className="space-y-4">
                        <button
                            onClick={() => onLogin('teacher')}
                            className="w-full flex items-center gap-4 p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-primary-500/50 transition-all duration-200 group cursor-pointer"
                        >
                            <div className="w-12 h-12 rounded-xl bg-primary-500/20 flex items-center justify-center group-hover:bg-primary-500/30 transition-colors">
                                <GraduationCap size={24} className="text-primary-400" />
                            </div>
                            <div className="text-left">
                                <p className="text-white font-semibold">Login as Teacher</p>
                                <p className="text-slate-400 text-sm">Upload videos & analyze engagement</p>
                            </div>
                        </button>

                        <button
                            onClick={() => onLogin('hod')}
                            className="w-full flex items-center gap-4 p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-primary-500/50 transition-all duration-200 group cursor-pointer"
                        >
                            <div className="w-12 h-12 rounded-xl bg-violet-500/20 flex items-center justify-center group-hover:bg-violet-500/30 transition-colors">
                                <ShieldCheck size={24} className="text-violet-400" />
                            </div>
                            <div className="text-left">
                                <p className="text-white font-semibold">Login as HOD</p>
                                <p className="text-slate-400 text-sm">Department-wide analytics overview</p>
                            </div>
                        </button>
                    </div>
                </div>

                <p className="text-center text-slate-500 text-xs mt-6">
                    Final Year Project &middot; Classroom Engagement System
                </p>
            </div>
        </div>
    )
}
