// 博客搜索功能
class BlogSearch {
    constructor() {
        this.searchIndex = [];
        this.searchInput = document.getElementById('search-input');
        this.searchResults = document.getElementById('search-results');
        this.init();
    }

    async init() {
        await this.loadSearchIndex();
        this.setupEventListeners();
        this.handleUrlParameters();
    }

    async loadSearchIndex() {
        try {
            const response = await fetch('/search-index.json');
            this.searchIndex = await response.json();
        } catch (error) {
            console.error('加载搜索索引失败:', error);
        }
    }

    setupEventListeners() {
        this.searchInput.addEventListener('input', this.debounce(this.performSearch.bind(this), 300));
        this.searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
        
        // 监听搜索结果链接点击事件
        document.addEventListener('click', this.handleResultClick.bind(this));
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    performSearch() {
        const query = this.searchInput.value.trim();
        
        if (!query) {
            this.showInitialState();
            this.updateUrl('');
            return;
        }

        const results = this.search(query);
        this.displayResults(results, query);
        this.updateUrl(query);
    }

    handleUrlParameters() {
        const urlParams = new URLSearchParams(window.location.search);
        const query = urlParams.get('q');
        
        if (query && query.trim()) {
            this.searchInput.value = query.trim();
            this.performSearch();
        }
    }

    updateUrl(query) {
        const url = new URL(window.location);
        
        if (query && query.trim()) {
            url.searchParams.set('q', query.trim());
        } else {
            url.searchParams.delete('q');
        }
        
        // 使用replaceState更新URL而不刷新页面
        window.history.replaceState({}, '', url.toString());
    }

    handleResultClick(event) {
        // 检查点击的是否是搜索结果链接
        const link = event.target.closest('.search-result-item a[data-return-url]');
        if (link) {
            // 保存返回URL到sessionStorage
            const returnUrl = link.getAttribute('data-return-url');
            sessionStorage.setItem('searchReturnUrl', returnUrl);
        }
    }

    search(query) {
        const searchTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 0);
        
        if (searchTerms.length === 0) {
            return [];
        }

        return this.searchIndex.map(post => {
            const score = this.calculateRelevance(post, searchTerms);
            return { post, score };
        })
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .map(item => item.post);
    }

    calculateRelevance(post, searchTerms) {
        let score = 0;
        const content = this.getSearchableContent(post).toLowerCase();

        searchTerms.forEach(term => {
            // 标题匹配（最高权重）
            if (post.title && post.title.toLowerCase().includes(term)) {
                score += 10;
            }
            
            // 副标题匹配（较高权重）
            if (post.subtitle && post.subtitle.toLowerCase().includes(term)) {
                score += 8;
            }
            
            // 标签匹配（中等权重）
            if (post.tags && Array.isArray(post.tags)) {
                const tagMatch = post.tags.some(tag => 
                    tag.toLowerCase().includes(term)
                );
                if (tagMatch) score += 6;
            }
            
            // 摘要匹配（较低权重）
            if (post.preview && post.preview.toLowerCase().includes(term)) {
                score += 4;
            }
            
            // 全文内容匹配（最低权重）
            if (content.includes(term)) {
                score += 2;
            }
        });

        return score;
    }

    getSearchableContent(post) {
        return [
            post.title || '',
            post.subtitle || '',
            post.preview || '',
            Array.isArray(post.tags) ? post.tags.join(' ') : ''
        ].join(' ');
    }

    displayResults(results, query) {
        if (results.length === 0) {
            this.searchResults.innerHTML = `
                <div class="no-results">
                    <p>没有找到与 "${this.escapeHtml(query)}" 相关的文章</p>
                </div>
            `;
            return;
        }

        const resultsHtml = results.map(post => this.createResultItem(post, query)).join('');
        this.searchResults.innerHTML = resultsHtml;
    }

    createResultItem(post, query) {
        const highlightedTitle = this.highlightText(post.title, query);
        const highlightedSubtitle = post.subtitle ? this.highlightText(post.subtitle, query) : '';
        const highlightedPreview = post.preview ? this.highlightText(post.preview.substring(0, 200), query) : '';
        
        const tagsHtml = Array.isArray(post.tags) ? 
            post.tags.map(tag => `<span class="search-tag">${this.escapeHtml(tag)}</span>`).join('') : '';

        // 构建包含搜索关键词的返回链接
        const returnUrl = query ? `/search/?q=${encodeURIComponent(query)}` : '/search/';
        
        return `
            <div class="search-result-item">
                <h3><a href="${post.url}" data-return-url="${returnUrl}">${highlightedTitle}</a></h3>
                ${post.subtitle ? `<h4>${highlightedSubtitle}</h4>` : ''}
                <div class="search-result-meta">
                    ${post.author ? `作者: ${this.escapeHtml(post.author)} | ` : ''}
                    发布于: ${post.date || '未知日期'}
                </div>
                <div class="search-result-preview">
                    ${highlightedPreview || '暂无摘要'}
                </div>
                ${tagsHtml ? `<div class="search-result-tags">${tagsHtml}</div>` : ''}
            </div>
        `;
    }

    highlightText(text, query) {
        if (!text || !query) return this.escapeHtml(text);
        
        const escapedText = this.escapeHtml(text);
        const searchTerms = query.toLowerCase().split(/\s+/);
        
        let highlightedText = escapedText;
        
        searchTerms.forEach(term => {
            if (term.length > 0) {
                const regex = new RegExp(`(${this.escapeRegex(term)})`, 'gi');
                highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
            }
        });
        
        return highlightedText;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    escapeRegex(text) {
        return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    showInitialState() {
        this.searchResults.innerHTML = `
            <div class="search-info">
                <p>输入关键词开始搜索</p>
            </div>
        `;
    }
}

// 页面加载完成后初始化搜索功能
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('search-input')) {
        new BlogSearch();
    }
});