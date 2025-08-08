export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string;
          email: string;
          full_name: string | null;
          avatar_url: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id: string;
          email: string;
          full_name?: string | null;
          avatar_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          email?: string;
          full_name?: string | null;
          avatar_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      documents: {
        Row: {
          id: string;
          user_id: string;
          title: string;
          file_name: string;
          file_type: string;
          file_size: number;
          content_preview: string | null;
          processing_status: 'pending' | 'processing' | 'completed' | 'failed';
          upload_path: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          title: string;
          file_name: string;
          file_type: string;
          file_size: number;
          content_preview?: string | null;
          processing_status?: 'pending' | 'processing' | 'completed' | 'failed';
          upload_path?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          title?: string;
          file_name?: string;
          file_type?: string;
          file_size?: number;
          content_preview?: string | null;
          processing_status?: 'pending' | 'processing' | 'completed' | 'failed';
          upload_path?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [
          {
            foreignKeyName: 'documents_user_id_fkey';
            columns: ['user_id'];
            referencedRelation: 'profiles';
            referencedColumns: ['id'];
          }
        ];
      };
      document_chunks: {
        Row: {
          id: string;
          document_id: string;
          chunk_index: number;
          content: string;
          metadata: Json;
          created_at: string;
        };
        Insert: {
          id?: string;
          document_id: string;
          chunk_index: number;
          content: string;
          metadata?: Json;
          created_at?: string;
        };
        Update: {
          id?: string;
          document_id?: string;
          chunk_index?: number;
          content?: string;
          metadata?: Json;
          created_at?: string;
        };
        Relationships: [
          {
            foreignKeyName: 'document_chunks_document_id_fkey';
            columns: ['document_id'];
            referencedRelation: 'documents';
            referencedColumns: ['id'];
          }
        ];
      };
      embeddings: {
        Row: {
          id: string;
          chunk_id: string;
          embedding: number[];
          created_at: string;
        };
        Insert: {
          id?: string;
          chunk_id: string;
          embedding: number[];
          created_at?: string;
        };
        Update: {
          id?: string;
          chunk_id?: string;
          embedding?: number[];
          created_at?: string;
        };
        Relationships: [
          {
            foreignKeyName: 'embeddings_chunk_id_fkey';
            columns: ['chunk_id'];
            referencedRelation: 'document_chunks';
            referencedColumns: ['id'];
          }
        ];
      };
      search_history: {
        Row: {
          id: string;
          user_id: string;
          query: string;
          results_count: number;
          created_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          query: string;
          results_count: number;
          created_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          query?: string;
          results_count?: number;
          created_at?: string;
        };
        Relationships: [
          {
            foreignKeyName: 'search_history_user_id_fkey';
            columns: ['user_id'];
            referencedRelation: 'profiles';
            referencedColumns: ['id'];
          }
        ];
      };
    };
    Views: {
      [_ in never]: never;
    };
    Functions: {
      [_ in never]: never;
    };
    Enums: {
      [_ in never]: never;
    };
    CompositeTypes: {
      [_ in never]: never;
    };
  };
}
