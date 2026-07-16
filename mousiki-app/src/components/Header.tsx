const Header = ({ trackCount }: { trackCount: number }) => (
  <header className="text-center mb-8">
    <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent">
      Mousiki
    </h1>
    <p className="text-gray-400 mt-2 max-w-xl mx-auto">
      A browser-based music recommendation playground. Add your own tracks and discover similar music — all computed client-side, zero backend.
    </p>
    <p className="text-sm text-gray-500 mt-1">{trackCount} tracks in catalog</p>
  </header>
)

export default Header
