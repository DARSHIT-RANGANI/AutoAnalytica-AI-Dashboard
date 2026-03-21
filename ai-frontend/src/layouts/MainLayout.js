import Sidebar from "./Sidebar";
import { useState } from "react";
import { motion } from "framer-motion";

export default function MainLayout({ children }) {
    const [darkMode, setDarkMode] = useState(true);

    return (
        <div className={darkMode ? "dark" : ""}>
            <div className="flex min-h-screen bg-gray-100 dark:bg-gray-950 text-black dark:text-white transition-all duration-300">
                <Sidebar />

                <div className="flex-1 p-6">
                    <div className="flex justify-end mb-4">
                        <button
                            onClick={() => setDarkMode(!darkMode)}
                            className="px-4 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition"
                        >
                            {darkMode ? "🌙 Dark" : "☀ Light"}
                        </button>
                    </div>

                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                    >
                        {children}
                    </motion.div>

                </div>
            </div>
        </div>
    );
}