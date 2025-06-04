import { useState } from 'react'

export const QueryParser = ({ onSubmit }) => {
    const [query, setQuery] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/language/parse_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error('Failed to parse query');
            }

            const result = await response.json();
            console.log('Query parse result:', result);
            
            if (onSubmit) {
                onSubmit(result);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="query-parser">
            <form onSubmit={handleSubmit} className="query-form">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter what you want to detect (e.g., 'Show me all vehicles and people wearing shirts')"
                    className="query-input"
                />
                <button 
                    type="submit" 
                    disabled={isLoading || !query.trim()}
                    className="query-submit"
                >
                    {isLoading ? 'Parsing...' : 'Apply Query'}
                </button>
            </form>
            
            {error && (
                <div className="error-message">
                    Error: {error}
                </div>
            )}
        </div>
    )
}