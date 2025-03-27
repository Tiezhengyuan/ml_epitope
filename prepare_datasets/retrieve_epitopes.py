import os
import json
import decimal
import mysql.connector

def export_large_table(outdir, prefix:str):
    config = {
        'host': 'localhost',
        'user': 'admin',
        'password': 'strong_password',
        'database': 'IEDB',
        # 'read_timeout': 86400,
        # 'connect_timeout': 30,
        # 'pool_reset_session': False,
    }
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor(buffered=True)

    # query = "SELECT * FROM view_epitope WHERE SUBSTRING(accession, 1, 1) = \'" + prefix + "\'"
    query = f"SELECT * FROM view_epitope WHERE accession LIKE \"{prefix}%\""
    print(query)
    cursor.execute(query)
    rows = cursor.fetchmany(10)
    rows = [{k: float(v) if isinstance(v, decimal.Decimal) else v 
            for k, v in row.items()} for row in rows]
    print(rows[0])

    outfile = os.path.join(outdir, f'epitope_{prefix}.json')
    with open(outfile, 'w') as f:
        json.dump(rows, f, indent=4)
    print(outfile)
    
    # Force clear (last resort)
    # cursor._connection.handle_unread_result() 
    # cursor.close()
    # conn.close()


json_dir = '/home/yuan/data/omics_data/epitope/mysql'
pool = [chr(i) for i in list(range(48,58)) + list(range(65, 91))]
print(pool)
for prefix in pool[10:]:
    try:
        export_large_table(json_dir, prefix)
    except Exception as e:
        print(f"Error from retrieval: {e}")
