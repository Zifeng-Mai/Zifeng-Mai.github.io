# Jekyll插件：自动生成搜索索引
require 'json'

module Jekyll
  class SearchIndexGenerator < Generator
    safe true
    priority :low

    def generate(site)
      # 创建搜索索引数据
      search_index = []
      
      # 遍历所有文章
      site.posts.docs.each do |post|
        # 提取文章信息
        post_data = {
          "title" => post.data['title'] || '',
          "subtitle" => post.data['subtitle'] || '',
          "tags" => post.data['tags'] || [],
          "preview" => post.data['preview'] || '',
          "url" => post.url,
          "date" => post.data['date'] ? post.data['date'].strftime('%Y-%m-%d') : '',
          "author" => post.data['author'] || site.config['title']
        }
        
        search_index << post_data
      end
      
      # 直接更新源目录中的搜索索引文件
      source_path = File.join(site.source, 'search-index.json')
      File.write(source_path, JSON.pretty_generate(search_index))
      
      Jekyll.logger.info "Search Index:", "Generated search index with #{search_index.size} posts"
    end
  end
end