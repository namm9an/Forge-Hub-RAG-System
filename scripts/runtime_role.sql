-- OPERATOR ONLY: run after migrations, with search_path selecting the app schema.
-- Creates a NOLOGIN permission role, never a password or application login.
-- Separately grant this role to a dedicated non-owner runtime LOGIN identity.
-- Keep that identity out of migration/owner/privileged roles.
-- Only named tables are granted: future views do not acquire implicit access.
DO $provision$
DECLARE
    permission_role TEXT := 'forge_hub_runtime';
    app_schema TEXT := current_schema();
    permission_oid OID;
    table_name TEXT;
    column_names TEXT;
BEGIN
    IF app_schema IS NULL THEN
        RAISE EXCEPTION 'Select an application schema before provisioning';
    END IF;
    SELECT oid INTO permission_oid FROM pg_roles WHERE rolname=permission_role;
    IF permission_oid IS NULL THEN
        EXECUTE format(
            'CREATE ROLE %I NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT',
            permission_role
        );
    ELSE
        IF EXISTS (
            SELECT 1 FROM pg_roles WHERE oid=permission_oid AND
            (rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls)
        ) OR EXISTS (SELECT 1 FROM pg_auth_members WHERE member=permission_oid)
          OR EXISTS (SELECT 1 FROM pg_class WHERE relowner=permission_oid)
          OR EXISTS (SELECT 1 FROM pg_namespace WHERE nspowner=permission_oid)
          OR EXISTS (SELECT 1 FROM pg_database WHERE datdba=permission_oid) THEN
            RAISE EXCEPTION 'Existing permission role must be a non-owner NOLOGIN role without memberships or elevated attributes';
        END IF;
    END IF;

    EXECUTE format('REVOKE ALL ON SCHEMA %I FROM %I', app_schema, permission_role);
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I', app_schema, permission_role);
    IF has_schema_privilege(permission_role, app_schema, 'CREATE')
       OR has_database_privilege(permission_role, current_database(), 'CREATE') THEN
        RAISE EXCEPTION 'Inherited schema/database CREATE remains; review PUBLIC grants separately';
    END IF;

    FOREACH table_name IN ARRAY ARRAY['sessions','documents','chunks','embeddings','chat_history','auth_rate_limits','schema_migrations']
    LOOP
        EXECUTE format('REVOKE ALL ON TABLE %I.%I FROM %I', app_schema, table_name, permission_role);
        SELECT string_agg(format('%I', attname), ',') INTO column_names
        FROM pg_attribute WHERE attrelid=format('%I.%I', app_schema, table_name)::regclass
            AND attnum>0 AND NOT attisdropped;
        EXECUTE format('REVOKE ALL (%s) ON TABLE %I.%I FROM %I',
                       column_names, app_schema, table_name, permission_role);
        IF table_name='schema_migrations' THEN
            -- Readiness reads versions; runtime must not change migration metadata.
            EXECUTE format('GRANT SELECT ON TABLE %I.%I TO %I', app_schema, table_name, permission_role);
        ELSE
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE %I.%I TO %I',
                           app_schema, table_name, permission_role);
        END IF;
    END LOOP;
    -- Managed tables use UUID or explicit keys. Revoke grants from prior provisioning.
    EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA %I FROM %I', app_schema, permission_role);
END;
$provision$;
