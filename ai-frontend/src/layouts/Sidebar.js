import { NavLink } from "react-router-dom";
import { useState } from "react";

export default function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);

    const linkClass =
        "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300";

    const activeClass = "bg-indigo-600 text-white";
    const inactiveClass = "hover:bg-gray-800 text-gray-300";

    return (
        <div
            className={`${collapsed ? "w-20" : "w-64"
                } bg-gray-900 min-h-screen p-4 border-r border-gray-800 transition-all duration-300`}
        >
            {/* Toggle Button */}
            <button
                onClick={() => setCollapsed(!collapsed)}
                className="mb-6 text-gray-400 hover:text-white"
            >
                ☰
            </button>

            {/* Logo */}
            {!collapsed && (
                <div className="mb-10 text-center">
                    <img
                        src="/logo.webp"
                        alt="logo"
                        className="w-20 mx-auto mb-2"
                    />
                    <h1 className="text-xl font-bold text-indigo-400">
                        AutoAnalytica
                    </h1>
                    <p className="text-xs text-gray-400">
                        CodeinYourself
                    </p>
                </div>
            )}

            {/* Navigation */}
            <nav className="flex flex-col gap-3">
                <NavLink
                    to="/"
                    className={({ isActive }) =>
                        `${linkClass} ${isActive ? activeClass : inactiveClass}`
                    }
                >
                    📊 {!collapsed && "Dashboard"}
                </NavLink>

                <NavLink
                    to="/upload"
                    className={({ isActive }) =>
                        `${linkClass} ${isActive ? activeClass : inactiveClass}`
                    }
                >
                    ⬆ {!collapsed && "Upload"}
                </NavLink>

                <NavLink
                    to="/models"
                    className={({ isActive }) =>
                        `${linkClass} ${isActive ? activeClass : inactiveClass}`
                    }
                >
                    🤖 {!collapsed && "Models"}
                </NavLink>

                <NavLink
                    to="/reports"
                    className={({ isActive }) =>
                        `${linkClass} ${isActive ? activeClass : inactiveClass}`
                    }
                >
                    📑 {!collapsed && "Reports"}
                </NavLink>
            </nav>
        </div>
    );
}