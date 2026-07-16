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
  Rock: 'from-red-500/20 to-orange-500/20 border-red-500/30',
  Pop: 'from-pink-500/20 to-rose-500/20 border-pink-500/30',
  Jazz: 'from-amber-500/20 to-yellow-500/20 border-amber-500/30',
  Electronic: 'from-cyan-500/20 to-blue-500/20 border-cyan-500/30',
  'Hip-Hop': 'from-yellow-500/20 to-orange-500/20 border-yellow-500/30',
  Classical: 'from-indigo-500/20 to-violet-500/20 border-indigo-500/30',
  'R&B': 'from-purple-500/20 to-fuchsia-500/20 border-purple-500/30',
  Country: 'from-amber-600/20 to-yellow-600/20 border-amber-600/30',
  Metal: 'from-gray-500/20 to-slate-500/20 border-gray-500/30',
  Reggae: 'from-green-500/20 to-emerald-500/20 border-green-500/30',
  Indie: 'from-teal-500/20 to-cyan-500/20 border-teal-500/30',
  Blues: 'from-blue-600/20 to-indigo-600/20 border-blue-600/30',
  Folk: 'from-stone-500/20 to-brown-500/20 border-stone-500/30',
  Latin: 'from-red-600/20 to-orange-600/20 border-red-600/30',
  Soul: 'from-violet-500/20 to-purple-500/20 border-violet-500/30',
  Funk: 'from-orange-500/20 to-red-500/20 border-orange-500/30',
  Punk: 'from-rose-600/20 to-red-600/20 border-rose-600/30',
  Ambient: 'from-sky-400/20 to-indigo-400/20 border-sky-400/30',
  'Pop Rock': 'from-orange-400/20 to-pink-400/20 border-orange-400/30',
  Alternative: 'from-lime-500/20 to-green-500/20 border-lime-500/30',
}

function getGenreColor(genres: string[]): string {
  if (genres.length === 0) return 'from-gray-500/20 to-gray-500/20 border-gray-500/30'
  return GENRE_COLORS[genres[0]] ?? 'from-gray-500/20 to-gray-500/20 border-gray-500/30'
}

const TrackCard = ({ track, selected, onClick, score, breakdown, onRemove, onSelect }: TrackCardProps) => {
  const color = getGenreColor(track.genres)

  return (
    <div
      className={`relative bg-gradient-to-br ${color} border rounded-xl p-4 transition-all duration-200 cursor-pointer
        ${selected ? 'ring-2 ring-purple-400 scale-[1.02] shadow-lg shadow-purple-500/20' : 'hover:scale-[1.02] hover:shadow-md'}`}
      onClick={() => { onClick?.(); onSelect?.(track) }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-white truncate">{track.title}</h3>
          <p className="text-sm text-gray-400 truncate">{track.artist}</p>
        </div>
        {score !== undefined && (
          <div className="text-right ml-2 shrink-0">
            <div className="text-lg font-bold text-purple-300">{(score * 100).toFixed(0)}%</div>
            <div className="text-xs text-gray-500">match</div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-1">
        {track.genres.map(g => <GenreBadge key={g} genre={g} />)}
      </div>

      {breakdown && (
        <div className="mt-3 pt-3 border-t border-white/10 text-xs text-gray-500 space-y-1">
          <div className="flex justify-between">
            <span>Genre</span>
            <span className="text-purple-300">{(breakdown.genreSimilarity * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between">
            <span>Artist</span>
            <span className="text-purple-300">{(breakdown.artistBoost * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between">
            <span>Title</span>
            <span className="text-purple-300">{(breakdown.textSimilarity * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {onRemove && (
        <button
          onClick={(e) => { e.stopPropagation(); onRemove(track.id) }}
          className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center rounded-full bg-black/30 text-gray-400 hover:text-red-400 hover:bg-black/50 text-xs"
        >
          ✕
        </button>
      )}
    </div>
  )
}

export default TrackCard
