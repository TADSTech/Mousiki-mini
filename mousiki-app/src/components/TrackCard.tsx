import { Play, X, Info } from 'lucide-react'
import type { Track } from '../engine/types'
import GenreBadge from './GenreBadge'

interface TrackCardProps {
  track: Track
  selected?: boolean
  onClick?: () => void
  score?: number
  breakdown?: { genreSimilarity: number; artistBoost: number; textSimilarity: number }
  hideActions?: boolean
  onRemove?: (id: string) => void
  onSelect?: (track: Track) => void
}

const GENRE_COLORS: Record<string, string> = {
  Rock: 'from-red-500/10 to-transparent',
  Pop: 'from-pink-500/10 to-transparent',
  Jazz: 'from-amber-500/10 to-transparent',
  Electronic: 'from-cyan-500/10 to-transparent',
  'Hip-Hop': 'from-yellow-500/10 to-transparent',
  Classical: 'from-indigo-500/10 to-transparent',
  'R&B': 'from-purple-500/10 to-transparent',
  Country: 'from-amber-600/10 to-transparent',
  Metal: 'from-gray-500/10 to-transparent',
  Reggae: 'from-green-500/10 to-transparent',
  Indie: 'from-teal-500/10 to-transparent',
  Blues: 'from-blue-600/10 to-transparent',
  Folk: 'from-stone-500/10 to-transparent',
  Latin: 'from-red-600/10 to-transparent',
  Soul: 'from-violet-500/10 to-transparent',
  Funk: 'from-orange-500/10 to-transparent',
  Punk: 'from-rose-600/10 to-transparent',
  Ambient: 'from-sky-400/10 to-transparent',
  'Pop Rock': 'from-orange-400/10 to-transparent',
  Alternative: 'from-lime-500/10 to-transparent',
}

function getGenreBg(genres: string[]): string {
  if (genres.length === 0) return 'from-gray-500/10 to-transparent'
  return GENRE_COLORS[genres[0]] ?? 'from-gray-500/10 to-transparent'
}

const TrackCard = ({ track, selected, onClick, score, breakdown, onRemove, onSelect }: TrackCardProps) => {
  const bgGradient = getGenreBg(track.genres)

  return (
    <div
      className={`relative group glass-card rounded-2xl p-5 overflow-hidden cursor-pointer bg-gradient-to-br ${bgGradient} 
        ${selected ? 'ring-2 ring-brand-500 scale-[1.02] shadow-[0_0_30px_-5px_rgba(59,130,246,0.3)] bg-gray-800/60' : 'hover:-translate-y-1'}`}
      onClick={() => { onClick?.(); onSelect?.(track) }}
    >
      {/* Remove Button */}
      {onRemove && (
        <button
          onClick={(e) => { e.stopPropagation(); onRemove(track.id) }}
          className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center rounded-full bg-gray-900/50 text-gray-400 opacity-0 group-hover:opacity-100 hover:bg-red-500/20 hover:text-red-400 transition-all duration-200 backdrop-blur-md"
        >
          <X className="w-4 h-4" />
        </button>
      )}

      <div className="flex items-start gap-4 mb-4">
        <div className="w-12 h-12 rounded-xl bg-gray-900 flex items-center justify-center shrink-0 border border-gray-800 shadow-inner group-hover:border-brand-500/30 transition-colors">
          <Play className="w-5 h-5 text-gray-400 group-hover:text-brand-400 transition-colors ml-1" />
        </div>
        
        <div className="flex-1 min-w-0 pr-8">
          <h3 className="font-bold text-gray-50 text-lg truncate group-hover:text-brand-300 transition-colors">{track.title}</h3>
          <p className="text-sm text-gray-400 truncate">{track.artist}</p>
        </div>
        
        {score !== undefined && (
          <div className="text-right shrink-0 bg-gray-900/50 px-3 py-1.5 rounded-lg border border-gray-800">
            <div className="text-xl font-black bg-gradient-to-br from-accent-400 to-brand-400 bg-clip-text text-transparent">
              {(score * 100).toFixed(0)}%
            </div>
            <div className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Match</div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-1.5">
        {track.genres.map(g => <GenreBadge key={g} genre={g} />)}
      </div>

      {breakdown && (
        <div className="mt-4 pt-4 border-t border-gray-800/50">
          <div className="flex items-center gap-1.5 mb-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
            <Info className="w-3.5 h-3.5" />
            Match Breakdown
          </div>
          <div className="space-y-1.5 text-xs text-gray-400">
            <div className="flex justify-between items-center bg-gray-900/30 rounded px-2 py-1">
              <span>Genre</span>
              <span className="font-mono text-accent-300">{(breakdown.genreSimilarity * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between items-center bg-gray-900/30 rounded px-2 py-1">
              <span>Artist</span>
              <span className="font-mono text-accent-300">{(breakdown.artistBoost * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between items-center bg-gray-900/30 rounded px-2 py-1">
              <span>Title</span>
              <span className="font-mono text-accent-300">{(breakdown.textSimilarity * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default TrackCard
