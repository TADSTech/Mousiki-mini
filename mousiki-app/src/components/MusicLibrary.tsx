import { useState } from 'react'
import { Search, Filter, LibraryBig } from 'lucide-react'
import type { Track, Genre } from '../engine/types'
import { ALL_GENRES } from '../engine/catalog'
import TrackCard from './TrackCard'

interface MusicLibraryProps {
  tracks: Track[]
  selectedTrack: Track | null
  onSelectTrack: (track: Track) => void
  onRemoveTrack: (id: string) => void
}

const MusicLibrary = ({ tracks, selectedTrack, onSelectTrack, onRemoveTrack }: MusicLibraryProps) => {
  const [search, setSearch] = useState('')
  const [genreFilter, setGenreFilter] = useState<Genre | 'All'>('All')

  const filtered = tracks.filter(t => {
    if (genreFilter !== 'All' && !t.genres.includes(genreFilter)) return false
    if (search) {
      const q = search.toLowerCase()
      if (!t.title.toLowerCase().includes(q) && !t.artist.toLowerCase().includes(q)) return false
    }
    return true
  })

  return (
    <div className="glass-panel rounded-3xl p-6 md:p-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2.5 bg-brand-500/10 rounded-xl text-brand-400">
          <LibraryBig className="w-6 h-6" />
        </div>
        <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-50 to-gray-300 bg-clip-text text-transparent">
          Your Library
        </h2>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 mb-8">
        <div className="relative flex-1 group">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 group-focus-within:text-brand-400 transition-colors" />
          <input
            type="text"
            placeholder="Search tracks or artists..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-gray-900/50 border border-gray-700/50 rounded-2xl text-gray-50 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500/50 transition-all shadow-inner"
          />
        </div>
        
        <div className="relative min-w-[160px] group">
          <Filter className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 group-focus-within:text-brand-400 transition-colors pointer-events-none" />
          <select
            value={genreFilter}
            onChange={e => setGenreFilter(e.target.value as Genre | 'All')}
            className="w-full pl-12 pr-10 py-3 bg-gray-900/50 border border-gray-700/50 rounded-2xl text-gray-50 appearance-none focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500/50 transition-all shadow-inner cursor-pointer"
          >
            <option value="All">All Genres</option>
            {ALL_GENRES.map(g => <option key={g} value={g}>{g}</option>)}
          </select>
          <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
            <svg width="12" height="8" viewBox="0 0 12 8" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 1.5L6 6.5L11 1.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {filtered.map(track => (
          <TrackCard
            key={track.id}
            track={track}
            selected={selectedTrack?.id === track.id}
            onSelect={onSelectTrack}
            onRemove={onRemoveTrack}
          />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center border-2 border-dashed border-gray-800 rounded-2xl bg-gray-900/20">
          <LibraryBig className="w-12 h-12 text-gray-600 mb-4" />
          <p className="text-gray-400 text-lg">No tracks match your filters.</p>
          <p className="text-gray-500 text-sm mt-1">Try adjusting your search or add more music!</p>
        </div>
      )}
    </div>
  )
}

export default MusicLibrary
