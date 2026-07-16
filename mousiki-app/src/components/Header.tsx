import { Music } from 'lucide-react'

const Header = ({ trackCount }: { trackCount: number }) => (
  <header className="relative text-center py-12 mb-8 glass-panel rounded-3xl overflow-hidden group">
    {/* Animated background glow */}
    <div className="absolute inset-0 bg-gradient-to-r from-brand-600/10 via-accent-600/10 to-brand-600/10 opacity-50 group-hover:opacity-100 transition-opacity duration-700" />
    
    <div className="relative z-10 flex flex-col items-center">
      <div className="w-16 h-16 mb-6 rounded-2xl bg-gradient-to-br from-brand-500 to-accent-600 p-0.5 shadow-lg shadow-brand-500/25">
        <div className="w-full h-full bg-gray-950 rounded-[14px] flex items-center justify-center">
          <Music className="w-8 h-8 text-brand-400" />
        </div>
      </div>

      <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight mb-4">
        <span className="bg-gradient-to-r from-brand-400 via-accent-400 to-brand-300 bg-clip-text text-transparent drop-shadow-sm">
          Mousiki
        </span>
      </h1>
      
      <p className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto px-4 font-light leading-relaxed">
        A premium browser-based music recommendation playground. Add your tracks and discover similar music seamlessly.
      </p>
      
      <div className="mt-8 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gray-900/80 border border-gray-800 text-sm font-medium text-gray-300 shadow-inner">
        <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse" />
        {trackCount} tracks in catalog
      </div>
    </div>
  </header>
)

export default Header
