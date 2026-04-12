import { LogOut } from 'lucide-react'

export default function Navbar({ title, role, onLogout }) {
    return (
        <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Logo & Title */}
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                            <span className="text-white font-bold text-sm">CE</span>
                        </div>
                        <div>
                            <h1 className="text-lg font-semibold text-slate-900">{title}</h1>
                            <p className="text-xs text-slate-500 -mt-0.5">Engagement Analytics</p>
                        </div>
                    </div>

                    {/* Right side */}
                    <div className="flex items-center gap-4">
                        <span className="text-sm text-slate-500 bg-slate-100 px-3 py-1 rounded-full font-medium capitalize">
                            {role}
                        </span>
                        <button
                            onClick={onLogout}
                            className="flex items-center gap-2 text-sm text-slate-500 hover:text-red-600 transition-colors cursor-pointer"
                        >
                            <LogOut size={16} />
                            Logout
                        </button>
                    </div>
                </div>
            </div>
        </nav>
    )
}
